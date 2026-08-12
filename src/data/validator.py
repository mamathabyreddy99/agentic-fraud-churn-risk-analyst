from __future__ import annotations
import pandas as pd

def validate_dataset(df: pd.DataFrame, target: str | None = None) -> dict:
    if df is None or df.empty:
        return {"valid": False, "errors": ["Dataset is empty."]}

    missing = df.isna().sum().sort_values(ascending=False)
    duplicate_count = int(df.duplicated().sum())
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    high_cardinality = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if df[c].nunique(dropna=True) > min(100, max(20, int(len(df) * 0.5)))
    ]

    result = {
        "valid": True,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": duplicate_count,
        "constant_columns": constant_cols,
        "high_cardinality_columns": high_cardinality,
        "missing_values": {k: int(v) for k, v in missing.items() if v > 0},
        "errors": [],
        "warnings": [],
    }

    if target:
        if target not in df.columns:
            result["valid"] = False
            result["errors"].append(f"Target '{target}' does not exist.")
        else:
            counts = df[target].value_counts(dropna=False)
            result["target_classes"] = {str(k): int(v) for k, v in counts.items()}
            result["target_unique"] = int(df[target].nunique(dropna=True))
            if result["target_unique"] != 2:
                result["warnings"].append("This project is designed for binary classification.")
            if len(counts) >= 2:
                smallest = counts.min()
                largest = counts.max()
                ratio = float(smallest / largest) if largest else 0.0
                result["minority_majority_ratio"] = ratio
                if ratio < 0.25:
                    result["warnings"].append("Target is strongly imbalanced.")

    return result
