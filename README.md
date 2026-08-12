# Agentic AI Fraud & Churn Risk Intelligence Platform

A portfolio-ready Streamlit application that combines deterministic machine learning with an agentic explanation layer.

## Features

- Upload fraud or churn CSV data
- Validate missing values, duplicates, constant columns, target balance, and high-cardinality columns
- Clean data interactively
- Detect severe class imbalance
- Compare Logistic Regression, Random Forest, and Gradient Boosting classifiers
- Handle imbalance with class weighting or SMOTE
- Evaluate Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and confusion matrix
- Generate row-level risk probabilities and risk bands
- Create downloadable flagged-cases and full risk reports
- Explain global risk factors with permutation importance
- Explain individual cases with model-agnostic local perturbation
- Ask natural-language questions through LangChain tools

## Design principle

All calculations, cleaning, training, metrics, scoring, and explainability are performed by Python.
The LLM only receives compact structured tool outputs and explains verified results.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

The AI assistant is optional. The deterministic ML application works without an API key.

## Tests

```bash
pytest -q
```

## Responsible use

A model risk score is not proof of fraud, misconduct, or customer intent. Use human review before consequential decisions.


## V2 portfolio upgrades
- Numerical and categorical EDA
- Automatic visualization recommendations
- Existing class-imbalance analysis with class weighting / SMOTE
- Model leaderboard with Accuracy, Precision, Recall, F1, ROC-AUC and PR-AUC
- Permutation feature importance
- Row-level risk explanations
- Downloadable cleaned data, leaderboard, full risk report and flagged cases
- LangChain/OpenAI tool-calling investigation layer

## Recommended demo flow
Upload CSV → inspect and clean → review imbalance → compare models → inspect risk leaderboard → explain a flagged case → ask the agent about model performance and risk factors.
