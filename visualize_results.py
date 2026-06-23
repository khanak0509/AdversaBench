import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12
})

def resolve_input(path):
    if path:
        return Path(path)
    for candidate in (Path("dataset_verified.json"), Path("dataset.json"), Path("output/dataset.json")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("dataset_verified.json not found")

def main():
    input_path = resolve_input(None)
    with open(input_path, 'r') as f:
        rows = json.load(f)
    

    data_heat = []
    for row in rows:
        cat = row["category"]
        for attempt in row.get("mutation_history", []):
            data_heat.append({
                "category": cat,
                "operator": attempt["operator"],
                "reward": attempt["reward"]
            })
    
    df_heat = pd.DataFrame(data_heat)
    pivot_heat = df_heat.pivot_table(values="reward", index="operator", columns="category", aggfunc="mean")
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_heat, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Operator Effectiveness (Mean Reward)")
    plt.tight_layout()
    plt.savefig("operator_heatmap.png")
    plt.close()
    

    max_iter = max(r.get("iterations", 0) for r in rows)
    survival_data = defaultdict(lambda: [0] * (max_iter + 1))
    
    for row in rows:
        cat = row["category"]
        iters = row.get("iterations", 0)
        failed = row.get("target_failed", False)
        

        for i in range(max_iter + 1):
            if not failed or iters > i:
                survival_data[cat][i] += 1
                survival_data["All"][i] += 1
                

    cat_counts = {cat: sum(1 for r in rows if r["category"] == cat) for cat in set(r["category"] for r in rows)}
    cat_counts["All"] = len(rows)
    
    plt.figure(figsize=(10, 6))
    for cat, counts in survival_data.items():
        percents = [c / cat_counts[cat] * 100 for c in counts]
        plt.plot(range(max_iter + 1), percents, marker='o', label=cat)
    
    plt.xlabel("Iteration")
    plt.ylabel("% Targets Surviving")
    plt.title("Target Model Survival by Iteration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("survival_curve.png")
    plt.close()
    

    disagree_data = []
    for row in rows:
        cat = row["category"]
        judges = row.get("judge_results", [])
        if not judges:
            continue
        fails = [j["failure_detected"] for j in judges]
        if all(fails):
            status = "Unanimous Fail"
        elif not any(fails):
            status = "Unanimous Pass"
        else:
            status = "Partial Disagreement"
        disagree_data.append({"category": cat, "status": status})
        
    df_disagree = pd.DataFrame(disagree_data)
    count_df = df_disagree.groupby(['category', 'status']).size().unstack(fill_value=0)
    
    count_df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='Set2')
    plt.title("Judge Disagreement by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Seeds")
    plt.xticks(rotation=0)
    plt.legend(title="Agreement Status")
    plt.tight_layout()
    plt.savefig("judge_disagreement.png")
    plt.close()


if __name__ == "__main__":
    main()
