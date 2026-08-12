APP_CONTEXT = {
    "training_result": None,
    "risk_report": None,
    "feature_importance": None,
    "dataset": None,
    "target": None,
}

def set_context(**kwargs):
    APP_CONTEXT.update(kwargs)
