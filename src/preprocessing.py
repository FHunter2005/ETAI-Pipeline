"""
Preprocessing: rule-based cleaning, training-only de-duplication, the feature/target split,
the leak-safe preprocessor and the development/test split.

`clean_dataset` (which also takes the missingness flags) and `split_features_target` learn nothing from the data, so they are safe on
any rows, including label-free data to predict. Everything learned from data (imputation,
encoding, scaling) lives in `build_preprocessor`, inside the model Pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler,
)
from category_encoders import CountEncoder


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies {column: {"min": ..., "max": ...}} domain rules (either bound optional) and sets
    violations to NaN, in place. Returns the number of violations per column.
    """
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def flag_inconsistent_bands(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies {label_column: {"score_column": ..., "bands": {label: [low, high]}}} rules: where
    the label disagrees with the band its score falls in, the label is set to NaN, in place
    (we cannot tell which of the two is wrong). Returns the number of disagreements per column.
    """
    report_rows = []
    for column, spec in rules.items():
        if column not in df.columns or spec["score_column"] not in df.columns:
            continue
        score = pd.to_numeric(df[spec["score_column"]], errors="coerce")
        expected = pd.Series(np.nan, index=df.index, dtype=object)
        for label, (low, high) in spec["bands"].items():
            expected[score.between(low, high)] = label
        disagreements = df[column].notna() & expected.notna() & (df[column] != expected)
        report_rows.append({"column": column, "disagreements": int(disagreements.sum())})
        df.loc[disagreements, column] = np.nan
    return pd.DataFrame(report_rows)


def reconstruct_from_total(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies {column: {"total": ..., "parts": [...]}} rules: where `column` is NaN, it is filled
    with total - sum(parts), in place, if all of those are known and the result is not negative.
    Returns the number of values filled per column.
    """
    report_rows = []
    for column, spec in rules.items():
        if column not in df.columns or spec["total"] not in df.columns:
            continue
        rebuilt = df[spec["total"]] - df[spec["parts"]].sum(axis=1, min_count=len(spec["parts"]))
        fill = df[column].isna() & rebuilt.notna() & (rebuilt >= 0)
        report_rows.append({"column": column, "filled": int(fill.sum())})
        df.loc[fill, column] = rebuilt[fill]
    return pd.DataFrame(report_rows)


def add_missingness_indicators(df: pd.DataFrame, missing_indicator_sources: list) -> pd.DataFrame:
    """Adds a `<col>_was_missing` flag for each listed column, before it is imputed."""
    out = df.copy()
    for col in missing_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict, missing_indicator_sources: list = ()) -> pd.DataFrame:
    """
    Applies the cleaning rules in `diagnostics_config`, in this order: placeholder tokens become
    NaN and category spellings are canonicalized; the `<col>_was_missing` flags are taken (so they
    mark values that were never recorded); domain-rule violations and label/score disagreements
    become NaN; parts are rebuilt from totals; redundant columns are dropped. Row-preserving and
    target-agnostic, so it runs unchanged on data to predict.
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")
    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), placeholder_tokens)

    out = add_missingness_indicators(out, missing_indicator_sources)

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))
    flag_inconsistent_bands(out, diagnostics_config.get("band_rules", {}))
    reconstruct_from_total(out, diagnostics_config.get("reconstruct_from_total", {}))

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    return out.drop(columns=columns_to_drop)


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """
    Training data only, before `split_dev_test`: drops exact duplicate rows and repeated ids
    (keeping the first). Never call it on data to predict: every row needs a prediction.
    """
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


def split_features_target(df: pd.DataFrame, data_config: dict):
    """
    Returns (X, y, extras). `extras` holds the columns kept out of the features for auditing
    (the sensitive attribute and COMPAS's own score). On label-free data `y` is None.
    """
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in always_drop]
    return df[feature_cols], y, extras


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
# sklearn's TargetEncoder cross-fits: a training row's encoding never uses its own label.
_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda seed: TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed)),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Builds the unfitted ColumnTransformer described by `preprocessing_config`: imputation and
    scaling for numeric columns, imputation and encoding for categorical ones, and the
    missingness flags passed through. Every encoder tolerates unseen categories.
    """
    imputation = preprocessing_config.get("imputation", {})
    scaler_factory = _SCALERS[preprocessing_config["scaler"]]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[preprocessing_config["encoder"]](preprocessing_config.get("random_state"))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])
    indicator_cols = [f"{c}_was_missing" for c in preprocessing_config.get("missing_indicator_sources", [])]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, preprocessing_config["numeric_features"]),
        ("categorical", categorical_pipeline, preprocessing_config["categorical_features"]),
        ("indicators", "passthrough", indicator_cols),
    ])


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """
    Stratified split into a development set, used for every decision, and a locked test set,
    used only for the final assessment. Returns X_dev, X_test, y_dev, y_test, extras_dev,
    extras_test, row-aligned.
    """
    return train_test_split(X, y, extras, test_size=test_size, random_state=random_state, stratify=y)