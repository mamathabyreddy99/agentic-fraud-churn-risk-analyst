from __future__ import annotations
import pandas as pd
from sklearn.inspection import permutation_importance

def global_permutation_importance(model, X_test, y_test, n_repeats: int = 5) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring="average_precision",
        n_jobs=-1,
    )

    return (
        pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

def explain_row(model, background: pd.DataFrame, row: pd.Series, top_k: int = 8) -> pd.DataFrame:
    row_df = pd.DataFrame([row.to_dict()])
    original_prob = float(model.predict_proba(row_df)[:, 1][0])
    details = []

    for feature in row.index:
        altered = row_df.copy()
        col = background[feature]

        if pd.api.types.is_numeric_dtype(col):
            baseline = col.median()
        else:
            mode = col.mode(dropna=True)
            baseline = mode.iloc[0] if not mode.empty else ""

        altered.loc[0, feature] = baseline
        changed_prob = float(model.predict_proba(altered)[:, 1][0])

        details.append({
            "feature": feature,
            "value": row[feature],
            "baseline": baseline,
            "risk_contribution": original_prob - changed_prob,
        })

    out = pd.DataFrame(details)
    out["abs_contribution"] = out["risk_contribution"].abs()
    return (
        out.sort_values("abs_contribution", ascending=False)
        .drop(columns="abs_contribution")
        .head(top_k)
        .reset_index(drop=True)
    )
