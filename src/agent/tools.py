from __future__ import annotations
from langchain.tools import tool
from .context import APP_CONTEXT
from src.ml.explainability import explain_row

@tool
def get_model_performance() -> dict:
    """Return the model leaderboard, best model, and confusion matrix."""
    result = APP_CONTEXT.get("training_result")
    if result is None:
        return {"error": "No model has been trained yet."}

    return {
        "best_model": result.best_model_name,
        "positive_class": result.positive_label,
        "leaderboard": result.leaderboard.round(4).to_dict(orient="records"),
        "confusion_matrix": result.confusion_matrix,
    }

@tool
def get_top_risk_factors(top_k: int = 8) -> dict:
    """Return the strongest global risk factors using permutation importance."""
    importance = APP_CONTEXT.get("feature_importance")
    if importance is None:
        return {"error": "Feature importance is not available yet."}

    return {
        "risk_factors": importance.head(top_k).round(5).to_dict(orient="records")
    }

@tool
def explain_flagged_case(row_id: str) -> dict:
    """Explain why one specific row received its model risk score."""
    result = APP_CONTEXT.get("training_result")
    report = APP_CONTEXT.get("risk_report")
    dataset = APP_CONTEXT.get("dataset")
    target = APP_CONTEXT.get("target")

    if result is None or report is None or dataset is None:
        return {"error": "Train a model and create a risk report first."}

    match = report.loc[report["row_id"].astype(str) == str(row_id)]
    if match.empty:
        return {"error": f"row_id '{row_id}' was not found."}

    try:
        source_index = int(row_id)
    except ValueError:
        return {"error": "row_id must be a numeric source-row identifier."}

    if source_index not in dataset.index:
        return {"error": "row_id does not exist in the source dataset."}

    X = dataset.drop(columns=[target]) if target in dataset.columns else dataset.copy()
    local = explain_row(result.best_model, X, X.loc[source_index])

    record = match.iloc[0]
    return {
        "row_id": str(row_id),
        "risk_probability": round(float(record["risk_probability"]), 5),
        "risk_band": str(record["risk_band"]),
        "top_local_factors": local.to_dict(orient="records"),
        "note": "Positive contribution means the observed value raised risk relative to the baseline replacement.",
    }

@tool
def get_flagged_cases(limit: int = 10) -> dict:
    """Return the highest-risk currently flagged rows."""
    report = APP_CONTEXT.get("risk_report")
    if report is None:
        return {"error": "Risk report is not available."}

    flagged = report.loc[report["risk_flag"]].head(limit)
    safe_cols = ["row_id", "risk_probability", "risk_band", "risk_flag"]
    return {"flagged_cases": flagged[safe_cols].round(5).to_dict(orient="records")}
