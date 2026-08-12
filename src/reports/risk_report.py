from __future__ import annotations
import pandas as pd

def build_risk_report(df: pd.DataFrame, model, target: str | None = None, threshold: float = 0.5) -> pd.DataFrame:
    X = df.drop(columns=[target]) if target and target in df.columns else df.copy()
    probabilities = model.predict_proba(X)[:, 1]

    report = df.copy()
    report.insert(0, "row_id", report.index.astype(str))
    report["risk_probability"] = probabilities
    report["risk_flag"] = probabilities >= threshold
    report["risk_band"] = pd.cut(
        probabilities,
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["Low", "Moderate", "High", "Critical"],
    ).astype(str)

    return report.sort_values("risk_probability", ascending=False).reset_index(drop=True)

def flagged_cases(report: pd.DataFrame) -> pd.DataFrame:
    return report.loc[report["risk_flag"]].copy().reset_index(drop=True)
