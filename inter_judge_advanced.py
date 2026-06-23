import json
import argparse
from pathlib import Path
from collections import defaultdict

JUDGES = [
    ("llama-70b", "Llama 70B"),
    ("cerebras-gpt-oss", "Cerebras GPT-OSS 120B"),
    ("qwen3", "Qwen3 32B"),
]

def resolve_input(path):
    if path:
        return Path(path)
    for candidate in (Path("dataset_verified.json"), Path("dataset.json"), Path("output/dataset.json")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("dataset_verified.json not found")

def main():
    parser = argparse.ArgumentParser(description="Advanced inter-judge analysis")
    parser.add_argument("--input", help="path to dataset_verified.json")
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    with open(input_path, 'r') as f:
        rows = json.load(f)




    confidence_stats = {k: {"fail_conf_sum": 0, "fail_count": 0, "pass_conf_sum": 0, "pass_count": 0} for k, _ in JUDGES}
    

    confusion_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for row in rows:
        results = {jr["judge"]: jr for jr in row.get("judge_results", [])}
        
        for j_key, _ in JUDGES:
            if j_key in results:
                res = results[j_key]
                conf = res.get("confidence", 0.0)
                if res.get("failure_detected"):
                    confidence_stats[j_key]["fail_conf_sum"] += conf
                    confidence_stats[j_key]["fail_count"] += 1
                else:
                    confidence_stats[j_key]["pass_conf_sum"] += conf
                    confidence_stats[j_key]["pass_count"] += 1
        

        for i, (j1, _) in enumerate(JUDGES):
            for j2, _ in JUDGES[i+1:]:
                if j1 in results and j2 in results:
                    res1, res2 = results[j1], results[j2]
                    if res1.get("failure_detected") and res2.get("failure_detected"):
                        type1 = res1.get("failure_type", "unknown")
                        type2 = res2.get("failure_type", "unknown")
                        confusion_matrix[f"{j1} vs {j2}"][type1][type2] += 1



if __name__ == "__main__":
    main()
