# AdversaBench

Automated LLM red-teaming benchmark. Takes seed prompts, mutates them adversarially, runs a weak target model, scores failures with a multi-judge panel, and exports a publication-ready failure dataset.

Built with **LangGraph** + **LangChain** (`ChatGroq`, `ChatOpenAI`, structured output, tool binding).

---

## Pipeline

```mermaid
flowchart LR
    S[seeds.json] --> A[Attacker]
    A -->|adversarial prompt| T[Target · Llama 8B]
    T --> J[Judges ×3]
    J -->|disagreement| M[Meta-judge · GPT-4o-mini]
    J -->|3/3 agree| SV[Save]
    M --> SV
    SV --> D[(dataset.json)]
    D --> C[dataset_clean.json]
    D --> V[dataset_verified.json]
```

**Loop:** if judges don't confirm a failure, the attacker mutates again (up to 5 iterations). Checkpoint resume so you can stop and continue.

---

## What we built

| Piece | What it does |
|-------|----------------|
| **Attacker** | 5 mutation operators + epsilon-greedy selection; escalates to GPT-4o-mini when Groq attacker can't break the target |
| **Target** | Groq Llama 3.1 8B — the model we're trying to break |
| **Judges** | 3-model panel (Groq 70B, Cerebras GPT-OSS 120B, Groq Qwen3) with Pydantic structured output |
| **Meta-judge** | GPT-4o-mini tiebreaker when judges disagree or error |
| **Tool-use** | 6 mock tools (`calculator`, `weather_api`, etc.) via LangChain `@tool` + `bind_tools` |
| **Datasets** | Tiered export — clean (unanimous) and verified (+ meta-judge) |
| **Audit** | GPT-4o-mini scores each clean row 1–5 for publication quality |

**30 seeds** — 10 reasoning, 10 instruction-following, 10 tool-use. Each has `expected_behavior` and `reference_answer` ground truth.

**5 operators:** `rephrase` · `inject_distractor` · `role_flip` · `constraint_add` · `jailbreak_wrap`

---

## Results (v3.1)

| | |
|---|---|
| Seeds run | **30 / 30** |
| Confirmed failures | **30** |
| Clean tier (3/3 judges) | **23** |
| Verified tier (+ meta-judge) | **7** |
| OpenAI audit (clean rows) | **23 / 23** scored ≥ 4/5 |
| Categories | reasoning 10 · instruction 10 · tool_use 10 |

---

## Quick start

```bash
pip install -r requirements.txt
```

Create `.env`:

```env
GROQ_API_KEY=...
CEREBRAS_API_KEY=...
OPENAI_API_KEY=...
```

```bash
python preflight.py              # sanity check
python main.py                   # full run (30 seeds)
python main.py --seed tool-002 --force   # single seed
python validate.py               # dataset QA
python audit.py                  # score clean tier
```

---

## Models

| Role | Model | Provider |
|------|-------|----------|
| Attacker | Llama 3.3 70B | Groq |
| Attacker escalation (iter 3+) | GPT-4o-mini | OpenAI |
| Target | Llama 3.1 8B Instant | Groq |
| Judge 1 | Llama 3.3 70B | Groq |
| Judge 2 | GPT-OSS 120B | Cerebras |
| Judge 3 | Qwen3 32B | Groq |
| Meta-judge | GPT-4o-mini | OpenAI |

All models configured in `config.yaml`. Swap a model there — no code changes needed.

---

## Output tiers

```
output/
├── dataset.json           # all 30 confirmed failures
├── dataset_clean.json     # 23 rows — unanimous 3/3 judge fail
├── dataset_verified.json  # 30 rows — clean + meta-judge verified
└── checkpoint.json        # resume progress
```

Each row includes: adversarial prompt, target response, judge verdicts, mutation history, consensus flag.

---

## Project structure

```
main.py          LangGraph pipeline
mutation.py      adversarial operators + attacker prompts
models.py        Pydantic schemas (JudgeVerdict, AttackerOutput, …)
tools.py         mock tools for tool_use seeds
config.yaml      models, paths, iteration limits
seeds.json       30 seeds with ground truth
audit.py         OpenAI publication-quality scoring
validate.py      dataset QA
preflight.py     pre-run smoke tests
```

---

## CLI

```bash
python main.py                        # run all seeds, skip checkpointed
python main.py --seed reasoning-001     # one seed
python main.py --seed tool-001 --force  # re-run, ignore checkpoint
python main.py --force-all              # re-run everything
```

---

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — pipeline orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — `ChatGroq`, `ChatOpenAI`
- [Pydantic](https://docs.pydantic.dev/) — typed judge/attacker outputs
- Groq · Cerebras · OpenAI — inference providers

---


## License

MIT
