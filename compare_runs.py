"""
Compares the runs in a config's scoreboard: `python compare_runs.py [config file]`.

Every run on the same data and metric is compared with the best one, on the same CV folds,
with the corrected paired t-test (`src.evaluate.corrected_paired_test`).
"""
import sys

import pandas as pd
import yaml

from src.evaluate import corrected_paired_test


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    metric = config["cv"]["selection_metric"]
    n_splits = config["cv"]["n_splits"]

    runs = pd.read_csv(config.get("output", {}).get("scoreboard", "results/runs.csv"), dtype={"run": str})
    runs = runs[(runs["data"] == config["data"]["path"]) & (runs["metric"] == metric)]
    runs = runs.sort_values("mean", ascending=False).reset_index(drop=True)
    fold_columns = [f"fold_{i}" for i in range(1, n_splits + 1)]
    best = runs.iloc[0]

    rows = []
    for _, run in runs.iterrows():
        test = corrected_paired_test(run[fold_columns], best[fold_columns], test_train_ratio=1 / (n_splits - 1))
        rows.append({"run": run["run"], "model": run["model"], "tuned": run["tuned"],
                     "feature_selection": run["feature_selection"], metric: run["mean"], "std": run["std"],
                     "diff_to_best": test["mean_diff"], "p_value": test["p_value"]})
    table = pd.DataFrame(rows)
    table.loc[0, "p_value"] = float("nan")                      # the best run is not compared with itself
    print(f"Runs on {config['data']['path']} ({metric}, {n_splits} folds), compared with the best run\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
