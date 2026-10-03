"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> diagnose/clean (week 3) -> split features/target
    -> leak-safe train/test split -> preprocess + train (week 3's encoder/scaler pair)
    -> evaluate (accuracy, fairness) -> save results
"""
import yaml
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_dev_test, split_features_target, build_preprocessor
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run
from sklearn.model_selection import cross_validate


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])
    print(f"clean_dataset:       {df_raw.shape} -> {df_clean.shape}   (same rows, same order: {df_clean.index.equals(df_raw.index)})")
    
    print("\nMissing values after cleaning:")
    print(df_clean.isna().sum().to_frame("missing after cleaning").T)
    print("\n")

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # leak-safe split: everything above this line is target/split-independent and may
    # see the whole dataset; everything below (imputation, encoding, scaling) is fit
    # only on the training fold, inside the Pipeline below
    X_train, X_test, y_train, y_test, extras_train, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    X_train_clean = drop_duplicate_rows(X_train, config["diagnostics"]["id_column"])
    print(f"drop_duplicate_rows: -> {X_train_clean.shape}   ({len(X_train) - len(X_train_clean)} duplicate rows removed -- training data only)")

    y_train = y_train.loc[X_train_clean.index]
    extras_train = extras_train.loc[X_train_clean.index]

    X_train = X_train_clean

    preprocessor = build_preprocessor(config["preprocessing"])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", build_model(config["model"])),
    ])
    cv_results = cross_validate(pipeline, X_train, y_train, cv=5, return_train_score=True)
    cv_train_acc = cv_results['train_score'].mean()
    cv_val_acc = cv_results['test_score'].mean()

    cv_report = (
        f"--- Cross-Validation (5 Folds) ---\n"
        f"CV Train accuracy (mean): {cv_train_acc:.3f}\n"
        f"CV Validation accuracy (mean): {cv_val_acc:.3f}\n"
        f"----------------------------------\n\n"
    )
    print(cv_report)
    pipeline.fit(X_train, y_train)

    # predict on both splits -- train accuracy vs. test accuracy is how we'll spot overfitting, not just how "good" the model looks
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    report = evaluate(y_train, y_train_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()