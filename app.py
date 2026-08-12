from __future__ import annotations

import os
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.data.validator import validate_dataset
from src.data.cleaner import clean_dataset
from src.ml.trainer import train_and_compare
from src.ml.explainability import global_permutation_importance, explain_row
from src.ml.feature_selection import drop_identifier_columns
from src.reports.risk_report import build_risk_report, flagged_cases
from src.utils.export import dataframe_to_csv_bytes
from src.agent.context import set_context
from src.analysis.eda import (
    numerical_summary,
    categorical_summary,
    recommend_visualizations,
)

load_dotenv()

st.set_page_config(
    page_title="Agentic AI Risk Analyst",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Agentic AI Fraud / Churn Risk Analyst")
st.caption(
    "Deterministic machine learning for scoring and explainability, "
    "with an optional tool-calling AI investigation layer."
)

# -------------------------
# Sidebar
# -------------------------

with st.sidebar:
    st.header("Dataset")

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    use_sample = st.checkbox(
        "Use sample dataset",
        value=uploaded is None,
    )

    st.divider()

    st.warning(
        "Model risk scores are decision-support signals. "
        "They should not be treated as proof of fraud or misconduct."
    )

# -------------------------
# Load dataset
# -------------------------

if uploaded is not None:
    raw_df = pd.read_csv(uploaded)

elif use_sample:
    raw_df = pd.read_csv("sample_data/risk_customers.csv")

else:
    st.info("Upload a CSV to begin.")
    st.stop()

# -------------------------
# 1. Dataset inspection
# -------------------------

st.subheader("1. Dataset inspection")

st.dataframe(
    raw_df.head(20),
    use_container_width=True,
)

default_target = (
    list(raw_df.columns).index("churn")
    if "churn" in raw_df.columns
    else len(raw_df.columns) - 1
)

target = st.selectbox(
    "Select binary target",
    options=list(raw_df.columns),
    index=default_target,
)

validation = validate_dataset(
    raw_df,
    target,
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Rows",
    validation.get("rows", 0),
)

c2.metric(
    "Columns",
    validation.get("columns", 0),
)

c3.metric(
    "Duplicates",
    validation.get("duplicate_rows", 0),
)

c4.metric(
    "Missing cells",
    sum(validation.get("missing_values", {}).values()),
)

with st.expander(
    "Validation details",
    expanded=True,
):
    st.json(validation)

# -------------------------
# 2. EDA
# -------------------------

st.subheader("2. Exploratory data analysis")

tab1, tab2 = st.tabs(
    [
        "Numerical summary",
        "Categorical summary",
    ]
)

with tab1:

    ns = numerical_summary(raw_df)

    if not ns.empty:
        st.dataframe(
            ns,
            use_container_width=True,
        )
    else:
        st.info(
            "No numerical columns."
        )

with tab2:

    cs = categorical_summary(raw_df)

    if not cs.empty:
        st.dataframe(
            cs,
            use_container_width=True,
        )
    else:
        st.info(
            "No categorical columns."
        )

st.markdown(
    "#### Recommended visualisations"
)

recommendations = recommend_visualizations(
    raw_df,
    target,
)

for rec in recommendations:

    st.write(
        f"**{rec['type']} — {rec['column']}**: "
        f"{rec['reason']}"
    )

# -------------------------
# 3. Cleaning
# -------------------------

st.subheader("3. Cleaning")

drop_duplicates = st.checkbox(
    "Remove duplicate rows",
    value=True,
)

drop_constants = st.checkbox(
    "Remove constant columns",
    value=True,
)

numeric_strategy = st.selectbox(
    "Numeric missing-value strategy",
    [
        "median",
        "mean",
    ],
)

clean_df = clean_dataset(
    raw_df,
    drop_duplicates=drop_duplicates,
    drop_constant_columns=drop_constants,
    numeric_strategy=numeric_strategy,
)

st.success(
    f"Working dataset: "
    f"{len(clean_df):,} rows × "
    f"{clean_df.shape[1]} columns"
)

st.download_button(
    "Download cleaned dataset",
    dataframe_to_csv_bytes(clean_df),
    file_name="cleaned_risk_dataset.csv",
    mime="text/csv",
)

if target not in clean_df.columns:

    st.error(
        "The target was removed during cleaning "
        "because it was constant."
    )

    st.stop()

# -------------------------
# 4. Target balance
# -------------------------

st.subheader("4. Target balance")

counts = (
    clean_df[target]
    .astype(str)
    .value_counts()
    .reset_index()
)

counts.columns = [
    "class",
    "count",
]

st.plotly_chart(
    px.bar(
        counts,
        x="class",
        y="count",
        title="Target class distribution",
    ),
    use_container_width=True,
)

strategy = st.radio(
    "Class-imbalance strategy",
    [
        "class_weight",
        "smote",
    ],
    format_func=lambda x: (
        "Class weighting"
        if x == "class_weight"
        else "SMOTE"
    ),
    horizontal=True,
)

threshold = st.slider(
    "Risk flag threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
)

# -------------------------
# Train models
# -------------------------

if st.button(
    "Train & compare models",
    type="primary",
    use_container_width=True,
):

    try:

        with st.spinner(
            "Training models..."
        ):

            st.session_state[
                "training_result"
            ] = train_and_compare(
                clean_df,
                target,
                imbalance_strategy=strategy,
            )

    except Exception as exc:

        st.exception(exc)

result = st.session_state.get(
    "training_result"
)

# -------------------------
# Results
# -------------------------

if result is not None:

    # -------------------------
    # 5. Leaderboard
    # -------------------------

    st.subheader(
        "5. Model leaderboard"
    )

    st.success(
        f"Selected best model: "
        f"{result.best_model_name}"
    )

    excluded_ids = getattr(result, "excluded_identifier_columns", [])

if excluded_ids:
    st.info(
        "Automatically excluded identifier columns from ML training: "
        + ", ".join(excluded_ids)
    )


    st.dataframe(
        result.leaderboard,
        use_container_width=True,
    )

    st.download_button(
        "Download model leaderboard",
        dataframe_to_csv_bytes(
            result.leaderboard
        ),
        file_name="model_leaderboard.csv",
        mime="text/csv",
    )

    # -------------------------
    # Confusion matrix
    # -------------------------

    cm = pd.DataFrame(
        result.confusion_matrix,
        index=[
            "Actual 0",
            "Actual 1",
        ],
        columns=[
            "Predicted 0",
            "Predicted 1",
        ],
    )

    st.markdown(
        "#### Confusion matrix"
    )

    st.dataframe(
        cm,
        use_container_width=True,
    )

    # -------------------------
    # ROC and PR curves
    # -------------------------

    curve1, curve2 = st.columns(2)

    with curve1:

        st.plotly_chart(
            px.line(
                result.roc_data,
                x="false_positive_rate",
                y="true_positive_rate",
                title="ROC curve",
            ),
            use_container_width=True,
        )

    with curve2:

        st.plotly_chart(
            px.line(
                result.pr_data,
                x="recall",
                y="precision",
                title="Precision–Recall curve",
            ),
            use_container_width=True,
        )

    # -------------------------
    # 6. Risk scoring
    # -------------------------

    st.subheader(
        "6. Risk scoring"
    )

    report = build_risk_report(
        clean_df,
        result.best_model,
        target=target,
        threshold=threshold,
    )

    flagged = flagged_cases(
        report
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Flagged cases",
        len(flagged),
    )

    m2.metric(
        "Flag rate",
        f"{len(flagged) / max(len(report), 1):.1%}",
    )

    m3.metric(
        "Highest score",
        f"{report['risk_probability'].max():.1%}",
    )

    st.dataframe(
        report[
            [
                "row_id",
                "risk_probability",
                "risk_band",
                "risk_flag",
            ]
        ].head(100),
        use_container_width=True,
    )

    d1, d2 = st.columns(2)

    with d1:

        st.download_button(
            "Download full risk report",
            dataframe_to_csv_bytes(
                report
            ),
            file_name="risk_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with d2:

        st.download_button(
            "Download flagged cases",
            dataframe_to_csv_bytes(
                flagged
            ),
            file_name="flagged_cases.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # -------------------------
    # 7. Global explainability
    # -------------------------

    st.subheader(
        "7. Global model explainability"
    )

    try:

        importance = (
            global_permutation_importance(
                result.best_model,
                result.X_test,
                result.y_test_encoded,
                n_repeats=5,
            )
        )

        st.dataframe(
            importance.head(15),
            use_container_width=True,
        )

        chart_df = (
            importance
            .head(15)
            .sort_values(
                "importance_mean"
            )
        )

        st.plotly_chart(
            px.bar(
                chart_df,
                x="importance_mean",
                y="feature",
                orientation="h",
                title=(
                    "Permutation feature "
                    "importance"
                ),
            ),
            use_container_width=True,
        )

    except Exception as exc:

        importance = None

        st.warning(
            "Could not compute permutation "
            f"importance: {exc}"
        )

    # -------------------------
    # 8. Local explanation
    # -------------------------

    st.subheader(
        "8. Why was this case flagged?"
    )

    case_ids = (
        report["row_id"]
        .astype(str)
        .head(200)
        .tolist()
    )

    selected_id = st.selectbox(
        "Select row_id",
        case_ids,
    )

    try:

        source_idx = int(
            selected_id
        )

        X_full = clean_df.drop(
            columns=[target]
        )

        X_full, _ = (
            drop_identifier_columns(
                X_full
            )
        )

        if source_idx in X_full.index:

            local = explain_row(
                result.best_model,
                X_full,
                X_full.loc[source_idx],
            )

            probability = float(
                report.loc[
                    report[
                        "row_id"
                    ].astype(str)
                    == selected_id,
                    "risk_probability",
                ].iloc[0]
            )

            st.metric(
                "Risk probability",
                f"{probability:.1%}",
            )

            st.dataframe(
                local,
                use_container_width=True,
            )

    except Exception as exc:

        st.warning(
            "Local explanation "
            f"unavailable: {exc}"
        )

    # -------------------------
    # Agent context
    # -------------------------

    set_context(
        training_result=result,
        risk_report=report,
        feature_importance=importance,
        dataset=clean_df,
        target=target,
    )

    # -------------------------
    # 9. Agent
    # -------------------------

    st.subheader(
        "9. Agentic AI investigation"
    )

    if os.getenv(
        "OPENAI_API_KEY"
    ):

        question = st.text_input(
            "Ask the risk analyst",
            placeholder=(
                "Why is row 12 high risk? "
                "What are the strongest "
                "risk factors?"
            ),
        )

        if (
            st.button(
                "Ask AI agent"
            )
            and question.strip()
        ):

            try:

                from src.agent.agent import (
                    ask_agent
                )

                with st.spinner(
                    "Agent is calling verified "
                    "analysis tools..."
                ):

                    st.write(
                        ask_agent(
                            question
                        )
                    )

            except Exception as exc:

                st.error(
                    str(exc)
                )

    else:

        st.info(
            "Add OPENAI_API_KEY to .env "
            "to enable the agent. "
            "Everything above works "
            "without an OpenAI key."
        )