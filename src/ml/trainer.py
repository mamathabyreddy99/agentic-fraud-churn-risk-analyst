from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from .feature_selection import drop_identifier_columns
from .preprocessing import build_preprocessor


# ---------------------------------------------------------
# TRAINING RESULT OBJECT
# ---------------------------------------------------------

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

    excluded_identifier_columns: list

    roc_data: pd.DataFrame
    pr_data: pd.DataFrame


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(
    y_true,
    probabilities,
    predictions,
) -> dict:

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
    }


# ---------------------------------------------------------
# TRAIN + COMPARE
# ---------------------------------------------------------

def train_and_compare(
    df: pd.DataFrame,
    target: str,
    imbalance_strategy: str = "class_weight",
    test_size: float = 0.25,
    random_state: int = 42,
):

    # -----------------------------------------------------
    # Prepare dataset
    # -----------------------------------------------------

    data = df.dropna(
        subset=[target]
    ).copy()

    X = data.drop(
        columns=[target]
    )

    # Remove ID-like columns such as customer_id
    X, excluded_ids = (
        drop_identifier_columns(X)
    )

    y_raw = (
        data[target]
        .astype(str)
    )

    if y_raw.nunique() != 2:

        raise ValueError(
            "Target must contain exactly two classes."
        )

    # -----------------------------------------------------
    # Encode target
    # -----------------------------------------------------

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        y_raw
    )

    positive_label = str(
        encoder.classes_[1]
    )

    # -----------------------------------------------------
    # Train / Test Split
    # -----------------------------------------------------

    class_counts = np.bincount(y)

    stratify_value = (
        y
        if (
            len(class_counts) == 2
            and class_counts.min() >= 2
        )
        else None
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_value,
    )

    # -----------------------------------------------------
    # Preprocessor
    # -----------------------------------------------------

    preprocessor = build_preprocessor(
        X_train
    )

    # -----------------------------------------------------
    # Models
    # -----------------------------------------------------

    models = {
        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=random_state,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=random_state,
            ),
    }

    # -----------------------------------------------------
    # Train each model
    # -----------------------------------------------------

    leaderboard_rows = []
    fitted_models = {}
    probability_results = {}

    for model_name, estimator in models.items():

        # ---------------------------------------------
        # SMOTE pipeline
        # ---------------------------------------------

        if imbalance_strategy == "smote":

            pipeline = ImbPipeline(
                steps=[
                    (
                        "preprocess",
                        clone(preprocessor),
                    ),
                    (
                        "smote",
                        SMOTE(
                            random_state=random_state
                        ),
                    ),
                    (
                        "model",
                        estimator,
                    ),
                ]
            )

        # ---------------------------------------------
        # Class-weight pipeline
        # ---------------------------------------------

        else:

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocess",
                        clone(preprocessor),
                    ),
                    (
                        "model",
                        estimator,
                    ),
                ]
            )

        # ---------------------------------------------
        # Fit
        # ---------------------------------------------

        pipeline.fit(
            X_train,
            y_train,
        )

        # ---------------------------------------------
        # Probability
        # ---------------------------------------------

        probabilities = (
            pipeline
            .predict_proba(X_test)[:, 1]
        )

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        metrics = calculate_metrics(
            y_test,
            probabilities,
            predictions,
        )

        leaderboard_rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

        fitted_models[
            model_name
        ] = pipeline

        probability_results[
            model_name
        ] = probabilities

    # -----------------------------------------------------
    # Leaderboard
    # -----------------------------------------------------

    leaderboard = pd.DataFrame(
        leaderboard_rows
    )

    leaderboard = (
        leaderboard
        .sort_values(
            by=[
                "pr_auc",
                "roc_auc",
                "f1",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Select best model
    # -----------------------------------------------------

    best_model_name = str(
        leaderboard.iloc[0][
            "model"
        ]
    )

    best_model = fitted_models[
        best_model_name
    ]

    best_probabilities = (
        probability_results[
            best_model_name
        ]
    )

    best_predictions = (
        best_probabilities >= 0.50
    ).astype(int)

    # -----------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_test,
        best_predictions,
    ).tolist()

    # -----------------------------------------------------
    # ROC CURVE
    # -----------------------------------------------------

    (
        false_positive_rate,
        true_positive_rate,
        roc_thresholds,
    ) = roc_curve(
        y_test,
        best_probabilities,
    )

    roc_data = pd.DataFrame(
        {
            "false_positive_rate":
                false_positive_rate,

            "true_positive_rate":
                true_positive_rate,

            "threshold":
                roc_thresholds,
        }
    )

    # -----------------------------------------------------
    # PRECISION-RECALL CURVE
    # -----------------------------------------------------

    (
        precision_values,
        recall_values,
        pr_thresholds,
    ) = precision_recall_curve(
        y_test,
        best_probabilities,
    )

    # precision_recall_curve returns one more
    # precision/recall value than thresholds.
    pr_data = pd.DataFrame(
        {
            "recall":
                recall_values,

            "precision":
                precision_values,
        }
    )

    # -----------------------------------------------------
    # Return all results
    # -----------------------------------------------------

    return TrainingResult(
        leaderboard=leaderboard,

        models=fitted_models,

        best_model_name=
            best_model_name,

        best_model=
            best_model,

        X_test=
            X_test,

        y_test_encoded=
            y_test,

        target_encoder=
            encoder,

        positive_label=
            positive_label,

        confusion_matrix=
            cm,

        excluded_identifier_columns=
            excluded_ids,

        roc_data=
            roc_data,

        pr_data=
            pr_data,
    )