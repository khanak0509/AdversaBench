
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os 
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

load_dotenv()
JUDGES = [
    ("llama-70b", "Llama 70B"),
    ("cerebras-gpt-oss", "Cerebras GPT-OSS 120B"),
    ("qwen3", "Qwen3 32B"),
]

PAIRS = [
    ("llama-70b", "cerebras-gpt-oss", "Llama 70B × Cerebras 120B"),
    ("llama-70b", "qwen3", "Llama 70B × Qwen3 32B"),
    ("cerebras-gpt-oss", "qwen3", "Cerebras 120B × Qwen3 32B"),
]


def resolve_input(path: str | None) -> Path:
    if path:
        return Path(path)
    for candidate in (Path("output/dataset.json"), Path("output copy/dataset.json")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("dataset.json not found — pass --input")


def verdict_label(failure_detected: bool) -> str:
    return "fail" if failure_detected else "pass"


def verdict_bit(failure_detected: bool) -> int:
    return 1 if failure_detected else 0


def kappa_interpretation(kappa: float) -> str:
    if kappa > 0.8:
        return "strong"
    if kappa >= 0.6:
        return "moderate"
    return "weak"


def extract_verdicts(row: dict) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for jr in row.get("judge_results", []):
        out[jr["judge"]] = jr["failure_detected"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Inter-judge analysis from dataset.json")
    parser.add_argument("--input", help="path to dataset.json")
    parser.add_argument("--output", default="judge_analysis.json", help="results json path")
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    rows = json.loads(input_path.read_text())
    n = len(rows)

    print(f"judge analysis, {n} rows\n")

    leniency_stats = {key: {"pass": 0, "fail": 0} for key, _ in JUDGES}
    row_verdicts: list[dict[str, bool]] = []

    for row in rows:
        verdicts = extract_verdicts(row)
        row_verdicts.append(verdicts)
        for key, _ in JUDGES:
            if key not in verdicts:
                continue
            if verdicts[key]:
                leniency_stats[key]["fail"] += 1
            else:
                leniency_stats[key]["pass"] += 1

    leniency_summary = {}
    print("leniency:")
    for key, label in JUDGES:
        s = leniency_stats[key]
        total = s["pass"] + s["fail"]
        rate = 100 * s["pass"] / total if total else 0.0
        leniency_summary[key] = {
            "label": label,
            "pass_votes": s["pass"],
            "fail_votes": s["fail"],
            "leniency_score": round(rate, 1),
        }
        print(f"  {key}: {s['pass']} pass, {s['fail']} fail ({rate:.0f}% lenient)")

    pairwise_summary = {}
    print("\npairwise:")
    for a_key, b_key, pair_label in PAIRS:
        a_bits, b_bits = [], []
        agree = 0
        counted = 0
        for verdicts in row_verdicts:
            if a_key not in verdicts or b_key not in verdicts:
                continue
            a_bit = verdict_bit(verdicts[a_key])
            b_bit = verdict_bit(verdicts[b_key])
            a_bits.append(a_bit)
            b_bits.append(b_bit)
            counted += 1
            if a_bit == b_bit:
                agree += 1

        agreement = 100 * agree / counted if counted else 0.0
        kappa = float(cohen_kappa_score(a_bits, b_bits)) if counted else 0.0
        interp = kappa_interpretation(kappa)
        pairwise_summary[f"{a_key}+{b_key}"] = {
            "label": pair_label,
            "agreement_rate": round(agreement, 1),
            "cohens_kappa": round(kappa, 3),
            "interpretation": interp,
            "rows_compared": counted,
        }
        print(f"  {a_key} + {b_key}: {agreement:.0f}% agree, kappa {kappa:.3f} ({interp})")

    categories: dict[str, list[bool]] = {}
    for row, verdicts in zip(rows, row_verdicts):
        cat = row["category"]
        bits = [verdict_bit(verdicts[k]) for k, _ in JUDGES if k in verdicts]
        disagree = len(set(bits)) > 1 if len(bits) >= 2 else False
        categories.setdefault(cat, []).append(disagree)

    category_summary = {}
    print("\nby category:")
    for cat in sorted(categories):
        flags = categories[cat]
        total = len(flags)
        disagreements = sum(flags)
        rate = 100 * disagreements / total if total else 0.0
        category_summary[cat] = {
            "total_seeds": total,
            "disagreements": disagreements,
            "disagreement_rate": round(rate, 1),
        }
        print(f"  {cat}: {disagreements}/{total} split ({rate:.0f}%)")

    out = {
        "input": str(input_path),
        "rows_analyzed": n,
        "leniency": leniency_summary,
        "pairwise": pairwise_summary,
        "category_disagreement": category_summary,
        "rows": [
            {
                "seed_id": row["seed_id"],
                "category": row["category"],
                "verdicts": {
                    k: verdict_label(v) for k, v in extract_verdicts(row).items()
                },
                "all_agree": len({verdict_bit(v) for v in extract_verdicts(row).values()}) <= 1,
            }
            for row in rows
        ],
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
