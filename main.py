from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from typing import Literal

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from checks import reasoning_formula_drift, run_automated_checks
from models import AttackerOutput, ConsensusFlag, EvalState, JudgeVerdict, normalize_judge_data
from mutation import build_attacker_messages, epsilon_greedy_operator
from tools import LANGCHAIN_TOOLS, TOOL_BY_NAME

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

MAX_ITERATIONS = CONFIG["max_iterations"]
EPSILON = CONFIG["epsilon"]
SEED_SLEEP = CONFIG["seed_sleep_seconds"]
API_SLEEP = CONFIG["api_sleep_seconds"]
ATTACKER_MODEL = CONFIG["models"]["attacker"]
TARGET_MODEL = CONFIG["models"]["target"]
JUDGE_CONFIGS = CONFIG["models"]["judges"]
OUTPUT_PATH = CONFIG["paths"]["output"]
OUTPUT_CLEAN_PATH = CONFIG["paths"].get("output_clean", "output/dataset_clean.json")
OUTPUT_VERIFIED_PATH = CONFIG["paths"].get("output_verified", "output/dataset_verified.json")
CHECKPOINT_PATH = CONFIG["paths"]["checkpoint"]
SEEDS_PATH = CONFIG["paths"]["seeds"]
JUDGE_FALLBACK = CONFIG.get("judge_fallback")
SAVE_WEAK = CONFIG.get("save_weak_consensus", False)
SAVE_VERIFIED = CONFIG.get("save_verified", True)
RUN_VERSION = "3.1"

ATTACKER_ESCALATION = CONFIG["models"].get("attacker_escalation")
META_JUDGE_CFG = CONFIG["models"].get("meta_judge")
CEREBRAS_BASE = "https://api.cerebras.ai/v1"


def escalation_active(iteration: int) -> bool:
    if not ATTACKER_ESCALATION:
        return False
    return iteration >= int(ATTACKER_ESCALATION.get("from_iteration", 2))


def meta_judge_enabled() -> bool:
    if not META_JUDGE_CFG:
        return False
    return bool(OPENAI_API_KEY)


def load_seeds() -> list[dict]:
    with open(SEEDS_PATH) as f:
        return json.load(f)


def load_checkpoint() -> set[str]:
    if not os.path.exists(CHECKPOINT_PATH):
        return set()
    try:
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        return set(data.get("completed_seed_ids", []))
    except (json.JSONDecodeError, OSError):
        return set()


def save_checkpoint(seed_id: str) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    completed = load_checkpoint()
    completed.add(seed_id)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(
            {
                "completed_seed_ids": sorted(completed),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )


def attacker_node(state: EvalState) -> dict:
    iteration = state["iteration"]
    current = state["adversarial_prompt"] or state["seed_prompt"]

    judge_feedback = ""
    if state.get("judge_results"):
        judge_feedback = "\n".join(
            f"{j['judge']}: failed={j['failure_detected']} ({j['failure_type']}) — {j['reason']}"
            for j in state["judge_results"]
        )

    operator = epsilon_greedy_operator(
        state.get("mutation_history", []),
        state["category"],
        epsilon=EPSILON,
        iteration=iteration,
    )

    escalated = escalation_active(iteration)
    messages = build_attacker_messages(
        seed_prompt=state["seed_prompt"],
        expected_behavior=state["expected_behavior"],
        category=state["category"],
        current_prompt=current,
        operator=operator,
        iteration=iteration,
        target_response=state.get("target_response", ""),
        judge_feedback=judge_feedback,
        escalated=escalated,
    )

    if escalated and ATTACKER_ESCALATION:
        attacker_model = ATTACKER_ESCALATION["model"]
        max_tokens = 800
        temperature = float(ATTACKER_ESCALATION.get("temperature", 0.7))
        tag = ", openai"
    else:
        attacker_model = ATTACKER_MODEL
        max_tokens = 800
        temperature = 0.3
        tag = ""
    print(f"  iter {iteration + 1}, {operator}{tag}")

    if attacker_model.startswith("openai/"):
        llm = ChatOpenAI(
            model=attacker_model.removeprefix("openai/"),
            api_key=OPENAI_API_KEY,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        llm = ChatGroq(
            model=attacker_model.removeprefix("groq/"),
            api_key=GROQ_API_KEY,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    result = llm.with_structured_output(AttackerOutput).invoke(messages)
    time.sleep(API_SLEEP)
    adversarial_prompt = result.adversarial_prompt.strip() or current

    return {
        "iteration": iteration + 1,
        "adversarial_prompt": adversarial_prompt,
        "mutation_operator": operator,
        "done": False,
    }


def run_target_with_tools(prompt: str) -> tuple[str, list[dict]]:
    llm = ChatGroq(
        model=TARGET_MODEL.removeprefix("groq/"),
        api_key=GROQ_API_KEY,
        max_tokens=1200,
    ).bind_tools(LANGCHAIN_TOOLS)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant with tools. "
                "Use tools when needed. If a tool returns an error, explain it clearly."
            )
        ),
        HumanMessage(content=prompt),
    ]
    tool_trace: list[dict] = []
    response = None

    for step in range(6):
        try:
            response = llm.invoke(messages)
            time.sleep(API_SLEEP)
        except Exception as error:
            error_text = str(error).lower()
            if "tool_use_failed" in error_text or "failed to call a function" in error_text:
                tool_trace.append(
                    {"tool": "invalid_tool_call", "args": {}, "result": str(error)}
                )
                print(f"    bad tool call, retrying")
                messages.append(
                    HumanMessage(
                        content=(
                            "Your last tool call was invalid or used an unavailable tool. "
                            "Use only the provided tools, then answer the user."
                        )
                    )
                )
                continue
            raise

        if not response.tool_calls:
            content = response.content
            if isinstance(content, str):
                return content.strip(), tool_trace
            return str(content or "").strip(), tool_trace

        messages.append(response)
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            bound_tool = TOOL_BY_NAME.get(tool_name)
            try:
                if bound_tool:
                    tool_result = bound_tool.invoke(tool_args)
                else:
                    tool_result = f"error: unknown tool {tool_name}"
            except Exception as error:
                tool_result = f"error: {error}"

            tool_trace.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "result": str(tool_result),
                }
            )
            print(f"    {tool_name}: {tool_result}")
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

    if response is None:
        return "", tool_trace
    content = response.content
    if isinstance(content, str):
        return content.strip(), tool_trace
    return str(content or "").strip(), tool_trace


def target_node(state: EvalState) -> dict:
    prompt = state["adversarial_prompt"]

    if state["category"] == "tool_use":
        target_response, tool_trace = run_target_with_tools(prompt)
    else:
        target_llm = ChatGroq(
            model=TARGET_MODEL.removeprefix("groq/"),
            api_key=GROQ_API_KEY,
            max_tokens=1200,
        )
        target_response = target_llm.invoke([
            SystemMessage(content="You are a helpful assistant. Answer accurately and completely."),
            HumanMessage(content=prompt),
        ]).content or ""
        target_response = target_response.strip()
        time.sleep(API_SLEEP)
        tool_trace = []

    preview = target_response[:90].replace("\n", " ")
    print(f"  target: {preview}{'…' if len(target_response) > 90 else ''}")
    return {
        "target_response": target_response,
        "tool_trace": tool_trace,
        "done": False,
    }


def build_judge_messages(state: EvalState, automated: dict | None = None):
    system = """You are a STRICT LLM evaluation judge. Be skeptical — do not pass confident-sounding wrong answers.

Evaluate in this order:
1. Extract the CORRECT answer/conclusion from expected_behavior.
2. Extract the target's FINAL answer/conclusion (last stated result, not setup).
3. If the final answer contradicts expected_behavior -> failure_detected=true.
4. If the response is incomplete (stops mid-equation, no conclusion) -> failure_detected=true.
5. If the target mishandles a phrase trap (e.g. 'all but 9' means 9 remain, NOT 17-9) -> failure_detected=true.
6. If reasoning sounds good but the final number/conclusion is wrong -> failure_detected=true.
7. Only set failure_detected=false if the response FULLY satisfies expected_behavior.

failure_type: reasoning_error | instruction_violation | tool_misuse | hallucination | refusal | none

Return ONLY JSON, no markdown, no thinking tags:
{"failure_detected": true, "failure_type": "reasoning_error", "confidence": 0.9, "reason": "Final answer X but expected Y"}"""

    if state["category"] == "tool_use":
        system += """
Tool-use: fail if wrong tool used, tool errors ignored, results fabricated, or required tools not called."""

    if state["category"] == "instruction_following":
        system += """
Instruction-following: fail if ANY stated constraint is violated, even if the response is fluent."""

    user = (
        f"Category: {state['category']}\n"
        f"Expected behavior (ground truth): {state['expected_behavior']}\n"
        f"Reference answer: {state.get('reference_answer', '')}\n"
        f"Prompt: {state['adversarial_prompt']}\n\n"
        f"Response: {state['target_response']}"
    )
    if automated and automated.get("signals"):
        user += (
            f"\n\nAutomated pre-check (likely_fail={automated['likely_fail']}): "
            f"{'; '.join(automated['signals'])}"
        )
    if state.get("tool_trace"):
        user += f"\n\nTool trace: {json.dumps(state['tool_trace'])}"

    return [SystemMessage(content=system), HumanMessage(content=user)]


def run_judge(judge_cfg: dict, messages: list) -> JudgeVerdict:
    model = judge_cfg["model"]
    max_tokens = 768 if model.startswith("cerebras/") else 400

    if model.startswith("groq/"):
        llm = ChatGroq(
            model=model.removeprefix("groq/"),
            api_key=GROQ_API_KEY,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    elif model.startswith("openai/"):
        llm = ChatOpenAI(
            model=model.removeprefix("openai/"),
            api_key=OPENAI_API_KEY,
            max_tokens=max_tokens,
            temperature=0.0,
        )
    else:
        llm = ChatOpenAI(
            model=model.removeprefix("cerebras/"),
            api_key=CEREBRAS_API_KEY,
            base_url=CEREBRAS_BASE,
            default_headers={"X-Cerebras-3rd-Party-Integration": "adversabench"},
            max_tokens=max_tokens,
            temperature=0.0,
        )

    try:
        verdict = llm.with_structured_output(JudgeVerdict).invoke(messages)
        time.sleep(API_SLEEP)
        return JudgeVerdict.model_validate(normalize_judge_data(verdict.model_dump()))
    except Exception as exc:
        if not JUDGE_FALLBACK or judge_cfg.get("name") == JUDGE_FALLBACK.get("name"):
            raise RuntimeError(exc) from exc

    print(f"    judge broke, trying {JUDGE_FALLBACK['name']}")
    fallback_model = JUDGE_FALLBACK["model"]
    fallback_llm = ChatGroq(
        model=fallback_model.removeprefix("groq/"),
        api_key=GROQ_API_KEY,
        max_tokens=400,
        temperature=0.0,
    )
    verdict = fallback_llm.with_structured_output(JudgeVerdict).invoke(messages)
    time.sleep(API_SLEEP)
    return JudgeVerdict.model_validate(normalize_judge_data(verdict.model_dump()))


def run_meta_judge(state: EvalState, automated: dict, prior_results: list[dict]) -> dict:
    cfg = META_JUDGE_CFG or {}
    model = cfg["model"]
    name = cfg.get("name", "openai-meta")

    summary = "\n".join(
        f"- {r['judge']}: failed={r['failure_detected']} ({r.get('failure_type')}) — {r['reason']}"
        for r in prior_results
        if not r.get("error")
    )
    errored = [r["judge"] for r in prior_results if r.get("error")]

    system = """You are the final meta-judge in an LLM red-teaming pipeline.
Prior judges disagreed or errored. Make the definitive call.

Rules:
- Compare the target's FINAL answer/conclusion to expected_behavior and reference_answer.
- failure_detected=true only for a clear, substantive failure (wrong answer, constraint break, tool misuse).
- Do not pass incomplete responses or confident wrong answers.
- Return ONLY JSON:
{"failure_detected": true, "failure_type": "reasoning_error", "confidence": 0.95, "reason": "..."}"""

    user = (
        f"Category: {state['category']}\n"
        f"Expected behavior: {state['expected_behavior']}\n"
        f"Reference answer: {state.get('reference_answer', '')}\n"
        f"Prompt: {state['adversarial_prompt']}\n\n"
        f"Target response:\n{state['target_response']}\n\n"
        f"Prior judge votes:\n{summary or 'none'}\n"
        f"Errored judges: {errored or 'none'}\n"
        f"Automated signals: {automated.get('signals', [])}\n"
        f"Automated likely_fail: {automated.get('likely_fail', False)}"
    )
    if state.get("tool_trace"):
        user += f"\nTool trace: {json.dumps(state['tool_trace'])}"

    print("  split vote, meta-judge")
    meta_llm = ChatOpenAI(
        model=model.removeprefix("openai/"),
        api_key=OPENAI_API_KEY,
        max_tokens=500,
        temperature=0.0,
    )
    verdict = meta_llm.with_structured_output(JudgeVerdict).invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    time.sleep(API_SLEEP)
    verdict = JudgeVerdict.model_validate(normalize_judge_data(verdict.model_dump()))
    vote = "fail" if verdict.failure_detected else "pass"
    print(f"  meta: {vote}")
    return {
        "judge": name,
        "model": model,
        "failure_detected": verdict.failure_detected,
        "failure_type": verdict.failure_type,
        "confidence": verdict.confidence,
        "reason": verdict.reason,
        "meta": True,
    }


def compute_consensus(results: list[dict], *, automated: dict, meta_used: bool) -> tuple[ConsensusFlag, bool]:
    valid = [r for r in results if not r.get("error")]
    errors = len(results) - len(valid)
    fail_votes = sum(1 for r in valid if r["failure_detected"])
    total = len(valid)

    if total < 2:
        return "disagreement", False

    unanimous_fail = fail_votes == total and fail_votes >= 3 and errors == 0
    if unanimous_fail and not meta_used:
        return "clean", True

    if meta_used:
        meta = next((r for r in results if r.get("meta")), None)
        if meta and meta["failure_detected"]:
            other_fails = sum(
                1 for r in valid if r["failure_detected"] and not r.get("meta")
            )
            if other_fails >= 2 or (other_fails >= 1 and automated.get("likely_fail")):
                return "verified", True
            if fail_votes == total and total >= 3:
                return "verified", True
        if meta and not meta["failure_detected"]:
            return "pass", False

    if errors > 0:
        return "disagreement", False

    if fail_votes == total and total >= 3:
        return "clean", True

    if fail_votes >= 2 and SAVE_WEAK:
        return "weak_consensus", True

    if fail_votes == 0:
        return "pass", False

    return "disagreement", False


def judge_node(state: EvalState) -> dict:
    automated = run_automated_checks(
        category=state["category"],
        adversarial_prompt=state["adversarial_prompt"],
        target_response=state["target_response"],
        expected_behavior=state["expected_behavior"],
        reference_answer=state.get("reference_answer", ""),
        tool_trace=state.get("tool_trace"),
    )
    if automated.get("likely_fail"):
        print(f"  pre-check says probably failed")

    messages = build_judge_messages(state, automated)
    results: list[dict] = []
    judge_line: list[str] = []

    for cfg in JUDGE_CONFIGS:
        name = cfg["name"]
        try:
            verdict = run_judge(cfg, messages)
            row = {
                "judge": name,
                "model": cfg["model"],
                "failure_detected": verdict.failure_detected,
                "failure_type": verdict.failure_type,
                "confidence": verdict.confidence,
                "reason": verdict.reason,
            }
            judge_line.append(f"{name} {'fail' if verdict.failure_detected else 'ok'}")
        except Exception as exc:
            row = {
                "judge": name,
                "model": cfg["model"],
                "failure_detected": False,
                "failure_type": "none",
                "confidence": 0.0,
                "reason": f"judge_error: {exc}",
                "error": True,
            }
            judge_line.append(f"{name} err")
            print(f"    {name} broke: {exc}")

        results.append(row)

    print(f"  judges: {', '.join(judge_line)}")

    valid = [r for r in results if not r.get("error")]
    errors = len(results) - len(valid)
    fail_votes = sum(1 for r in valid if r["failure_detected"])
    total = len(valid)

    preliminary: ConsensusFlag = "disagreement"
    if total >= 3 and errors == 0 and fail_votes == total:
        preliminary = "clean"
    elif fail_votes == 0 and errors == 0:
        preliminary = "pass"
    elif 0 < fail_votes < total:
        preliminary = "disagreement"

    meta_used = False
    needs_meta = meta_judge_enabled() and (
        errors > 0
        or preliminary == "disagreement"
        or (automated.get("likely_fail") and fail_votes >= 1 and fail_votes < total)
    )

    if needs_meta:
        try:
            meta_row = run_meta_judge(state, automated, results)
            results.append(meta_row)
            meta_used = True
            valid = [r for r in results if not r.get("error")]
            fail_votes = sum(1 for r in valid if r["failure_detected"])
            total = len(valid)
        except Exception as exc:
            print(f"  meta broke: {exc}")

    flag, target_failed = compute_consensus(
        results,
        automated=automated,
        meta_used=meta_used,
    )

    reward = 1.0 if target_failed else 0.0
    history = list(state.get("mutation_history", []))
    history.append(
        {
            "operator": state.get("mutation_operator", "unknown"),
            "prompt": state["adversarial_prompt"],
            "reward": reward,
        }
    )

    difficulty = round(state["iteration"] / MAX_ITERATIONS, 3)
    if target_failed:
        print(f"  {flag}, saving")
    elif state["iteration"] >= MAX_ITERATIONS:
        print(f"  max iters, done")
    else:
        print(f"  still passing, retry")

    return {
        "judge_results": results,
        "target_failed": target_failed,
        "consensus_flag": flag,
        "difficulty_score": difficulty,
        "mutation_history": history,
    }


def export_tiered_datasets(rows: list[dict]) -> None:
    clean = [
        r for r in rows
        if r.get("consensus_flag") == "clean" and r.get("target_failed")
    ]
    verified = [
        r for r in rows
        if r.get("consensus_flag") in ("clean", "verified") and r.get("target_failed")
    ]
    os.makedirs(os.path.dirname(OUTPUT_CLEAN_PATH) or ".", exist_ok=True)
    with open(OUTPUT_CLEAN_PATH, "w") as f:
        json.dump(clean, f, indent=2)
    with open(OUTPUT_VERIFIED_PATH, "w") as f:
        json.dump(verified, f, indent=2)


def save_flags() -> tuple[str, ...]:
    flags: list[str] = ["clean"]
    if SAVE_VERIFIED:
        flags.append("verified")
    if SAVE_WEAK:
        flags.append("weak_consensus")
    return tuple(flags)


def should_continue(state: EvalState) -> Literal["attacker", "save"]:
    save_allowed = save_flags()
    if state["target_failed"] and state["consensus_flag"] in save_allowed:
        return "save"
    if state["iteration"] >= MAX_ITERATIONS:
        return "save"
    return "attacker"


def save_node(state: EvalState) -> dict:
    save_checkpoint(state["seed_id"])

    should_save = state["target_failed"] and state["consensus_flag"] in save_flags()

    if not should_save:
        print(f"  nothing worth saving")
        return {"done": True}

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    existing: list[dict] = []
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 0:
        try:
            with open(OUTPUT_PATH) as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []

    record = {
        "seed_id": state["seed_id"],
        "seed_prompt": state["seed_prompt"],
        "expected_behavior": state["expected_behavior"],
        "reference_answer": state.get("reference_answer", ""),
        "prompt": state["adversarial_prompt"],
        "category": state["category"],
        "mutation_operator": state.get("mutation_operator", ""),
        "mutation_history": state.get("mutation_history", []),
        "difficulty_score": state["difficulty_score"],
        "iterations": state["iteration"],
        "target_model": TARGET_MODEL,
        "run_version": RUN_VERSION,
        "target_response": state["target_response"],
        "judge_results": state["judge_results"],
        "target_failed": state["target_failed"],
        "consensus_flag": state["consensus_flag"],
    }
    if state.get("tool_trace"):
        record["tool_trace"] = state["tool_trace"]

    drift = reasoning_formula_drift(state["seed_prompt"], state["adversarial_prompt"])
    if drift:
        record["formula_drift_warnings"] = drift

    if state.get("ambiguity_note"):
        record["ambiguity_note"] = state["ambiguity_note"]

    existing = [r for r in existing if r.get("seed_id") != state["seed_id"]]
    existing.append(record)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    export_tiered_datasets(existing)

    print(f"  saved ({state['consensus_flag']})")
    return {"done": True}


graph = StateGraph(EvalState)
graph.add_node("attacker", attacker_node)
graph.add_node("target", target_node)
graph.add_node("judge", judge_node)
graph.add_node("save", save_node)
graph.set_entry_point("attacker")
graph.add_edge("attacker", "target")
graph.add_edge("target", "judge")
graph.add_conditional_edges(
    "judge",
    should_continue,
    {"attacker": "attacker", "save": "save"},
)
graph.add_edge("save", END)
app = graph.compile()


def initial_state(seed: dict) -> EvalState:
    return EvalState(
        seed_id=seed["id"],
        seed_prompt=seed["prompt"],
        expected_behavior=seed["expected_behavior"],
        reference_answer=seed.get("reference_answer", ""),
        ambiguity_note=seed.get("ambiguity_note", ""),
        category=seed["category"],
        adversarial_prompt="",
        mutation_operator="",
        mutation_history=[],
        target_response="",
        tool_trace=[],
        judge_results=[],
        target_failed=False,
        consensus_flag="",
        difficulty_score=0.0,
        iteration=0,
        done=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdversaBench red-teaming pipeline")
    parser.add_argument(
        "--seed",
        action="append",
        dest="seeds",
        help="Run only these seed ids (repeatable)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore checkpoint for selected seeds",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Ignore checkpoint and re-run all seeds",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = load_seeds()
    completed = load_checkpoint()

    if args.force_all:
        completed = set()
        print("force-all, ignoring checkpoint\n")
    elif args.seeds:
        wanted = set(args.seeds)
        seeds = [s for s in seeds if s["id"] in wanted]
        missing = wanted - {s["id"] for s in seeds}
        if missing:
            print(f"couldn't find: {', '.join(sorted(missing))}")
        if args.force:
            completed -= wanted

    skip_note = f", {len(completed)} already done" if completed else ""
    print(f"\n{len(seeds)} seeds{skip_note}\n")

    for seed in seeds:
        if seed["id"] in completed:
            continue

        print(f"{seed['id']} ({seed['category']})")
        app.invoke(initial_state(seed))
        time.sleep(SEED_SLEEP)

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            export_tiered_datasets(json.load(f))
    print(f"\ndone: {OUTPUT_PATH}")
