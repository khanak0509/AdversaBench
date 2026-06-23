import json
import argparse
from pathlib import Path
from collections import defaultdict

def resolve_input(path):
    if path:
        return Path(path)
    for candidate in (Path("dataset_verified.json"), Path("dataset.json"), Path("output/dataset.json")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("dataset_verified.json not found")

def main():
    parser = argparse.ArgumentParser(description="Operator Selection Ablation")
    parser.add_argument("--input", help="path to dataset_verified.json")
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    with open(input_path, 'r') as f:
        rows = json.load(f)


    operator_rewards = defaultdict(list)
    

    transition_success = defaultdict(lambda: {"attempts": 0, "successes": 0})

    for row in rows:
        history = row.get("mutation_history", [])
        prev_op = None
        prev_reward = None
        
        for attempt in history:
            op = attempt["operator"]
            reward = attempt["reward"]
            operator_rewards[op].append(reward)
            
            if prev_op is not None and prev_reward == 0.0:

                transition_success[f"{prev_op} -> {op}"]["attempts"] += 1
                if reward > 0:
                    transition_success[f"{prev_op} -> {op}"]["successes"] += 1
            
            prev_op = op
            prev_reward = reward



if __name__ == "__main__":
    main()
