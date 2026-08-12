import pandas as pd
from src.data.validator import validate_dataset
from src.data.cleaner import clean_dataset
from src.ml.trainer import train_and_compare
from src.reports.risk_report import build_risk_report

def make_df(n=160):
    rows = []
    for i in range(n):
        risky = i % 8 == 0
        rows.append({
            "tenure_months": i % 48,
            "monthly_charge": 115 if risky else 55 + (i % 20),
            "support_tickets": 6 if risky else i % 3,
            "contract": "month-to-month" if risky else "annual",
            "churn": "Yes" if risky else "No",
        })
    return pd.DataFrame(rows)

def test_validation_binary_target():
    result = validate_dataset(make_df(), "churn")
    assert result["valid"] is True
    assert result["target_unique"] == 2

def test_cleaner_removes_exact_duplicate():
    df = make_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    expected_rows = len(df.drop_duplicates())
    cleaned = clean_dataset(df)
    assert len(cleaned) == expected_rows
    assert cleaned.duplicated().sum() == 0

def test_training_and_risk_report():
    df = make_df()
    result = train_and_compare(df, "churn", imbalance_strategy="class_weight")
    assert len(result.leaderboard) == 3
    report = build_risk_report(df, result.best_model, target="churn")
    assert "risk_probability" in report.columns
    assert report["risk_probability"].between(0, 1).all()
