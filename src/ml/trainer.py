from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .preprocessing import build_preprocessor

@dataclass
class TrainingResult:
    leaderboard: pd.DataFrame
    models: dict
    best_model_name: str
    best_model: object
    X_test: pd.DataFrame
    y_test_encoded: np.ndarray
    target_encoder: LabelEncoder
    positive_label: str
    confusion_matrix: list

def _metrics(y_true, prob, pred):
    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = average_precision_score(y_true, prob)
    except ValueError:
        metrics["pr_auc"] = float("nan")
    return metrics

def train_and_compare(
    df: pd.DataFrame,
    target: str,
    imbalance_strategy: str = "class_weight",
    test_size: float = 0.25,
    random_state: int = 42,
):
    data = df.dropna(subset=[target]).copy()
    X = data.drop(columns=[target])
    y_raw = data[target].astype(str)

    if y_raw.nunique() != 2:
        raise ValueError("Target must contain exactly two classes.")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    positive_label = str(encoder.classes_[1])

    bincount = np.bincount(y)
    stratify = y if len(bincount) == 2 and bincount.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    preprocessor = build_preprocessor(X_train)

    model_defs = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }

    fitted = {}
    rows = []

    for name, estimator in model_defs.items():
        if imbalance_strategy == "smote":
            pipe = ImbPipeline([
                ("preprocess", clone(preprocessor)),
                ("smote", SMOTE(random_state=random_state)),
                ("model", estimator),
            ])
        else:
            pipe = Pipeline([
                ("preprocess", clone(preprocessor)),
                ("model", estimator),
            ])

        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        rows.append({"model": name, **_metrics(y_test, prob, pred)})
        fitted[name] = pipe

    leaderboard = (
        pd.DataFrame(rows)
        .sort_values(["pr_auc", "roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )

    best_name = str(leaderboard.iloc[0]["model"])
    best_model = fitted[best_name]
    best_pred = (best_model.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

    return TrainingResult(
        leaderboard=leaderboard,
        models=fitted,
        best_model_name=best_name,
        best_model=best_model,
        X_test=X_test,
        y_test_encoded=y_test,
        target_encoder=encoder,
        positive_label=positive_label,
        confusion_matrix=confusion_matrix(y_test, best_pred).tolist(),
    )
