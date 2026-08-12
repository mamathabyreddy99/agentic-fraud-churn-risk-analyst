from __future__ import annotations
import pandas as pd

def clean_dataset(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    drop_constant_columns: bool = True,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> pd.DataFrame:
    out = df.copy()

    if drop_duplicates:
        out = out.drop_duplicates().copy()

    if drop_constant_columns:
        constant_cols = [c for c in out.columns if out[c].nunique(dropna=False) <= 1]
        if constant_cols:
            out = out.drop(columns=constant_cols)

    for c in out.select_dtypes(include="number").columns:
        if out[c].isna().any():
            fill = out[c].mean() if numeric_strategy == "mean" else out[c].median()
            out[c] = out[c].fillna(fill)

    for c in out.select_dtypes(exclude="number").columns:
        if out[c].isna().any():
            mode = out[c].mode(dropna=True)
            fill = mode.iloc[0] if categorical_strategy == "mode" and not mode.empty else "Unknown"
            out[c] = out[c].fillna(fill)

    return out.reset_index(drop=True)
