"""Saving each run's report and fitted model to disk."""
import os
from datetime import datetime

import joblib


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs

def save_run(results_dir: str, config: dict, report_text: str, model=None):
    """
    Writes the run's settings and full report to `results_dir/run_<timestamp>.txt` and, if
    given, the fitted `model` to `results_dir/model_<timestamp>.joblib` (the folder is created
    if needed). Returns (report_path, model_path); model_path is None without a model.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")

    cv = config["cv"]
    tuning = config.get("tuning", {})
    tuning_line = (f"Optuna, {tuning['n_trials']} trials x {tuning['n_splits']} folds, nested CV  "
                   f"random_state={tuning['random_state']}" if tuning.get("enabled") else "off")
    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  "
        f"scaler={config['preprocessing']['scaler']}\n"
        f"CV: {cv['n_splits']} stratified folds  shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}  metric={cv.get('scoring', 'accuracy')}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  (not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    model_path = None
    if model is not None:
        model_path = os.path.join(results_dir, f"model_{timestamp}.joblib")
        joblib.dump(model, model_path)
    return path, model_path