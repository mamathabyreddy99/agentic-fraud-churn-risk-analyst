# Agentic AI Fraud & Churn Risk Analyst

An agentic risk-analysis platform for exploring customer datasets, training classification models, generating risk scores, explaining predictions, and investigating model results through natural-language AI.

**Live Application:** Coming Soon  
**GitHub:** https://github.com/mamathabyreddy99/agentic-fraud-churn-risk-analyst  
**Developer:** Mamatha Byreddy  
**LinkedIn:** https://www.linkedin.com/in/byreddy-mamatha-296a221a8

---

## Project Overview

Agentic AI Fraud & Churn Risk Analyst combines a Streamlit frontend, Pandas-based data processing, Plotly visualisation, scikit-learn machine-learning pipelines, imbalanced-learning techniques, model explainability, and a LangChain tool-calling agent.

Users can upload customer or transaction CSV data and move through an end-to-end risk-analysis workflow without manually writing data-analysis or machine-learning code.

The project focuses specifically on **binary risk classification problems**, such as:

* customer churn prediction;
* fraud-risk analysis;
* customer risk scoring;
* high-risk case identification.

A key architectural decision is separating deterministic machine-learning operations from the language-model layer.

Python functions perform data processing, model training, evaluation, scoring, and explainability. The AI agent receives structured results through controlled tools and uses those results to answer natural-language questions.

---

# Features

## Dataset Overview

* CSV upload
* Built-in sample dataset
* Dataset preview
* Row and column counts
* Column data types
* Missing-value detection
* Duplicate-row detection
* Constant-column detection
* Binary-target validation
* Automatic identifier-column detection for model training

---

## Exploratory Data Analysis

### Numerical Analysis

The application generates descriptive summaries for numerical features, including:

* count;
* mean;
* standard deviation;
* minimum;
* quartiles;
* maximum.

### Categorical Analysis

Categorical features can be inspected using:

* unique-value counts;
* most frequent category;
* category frequency;
* categorical summaries.

### Visualisation Recommendations

The application examines available columns and recommends useful visualisations based on feature type and target selection.

The Streamlit interface uses interactive Plotly visualisations.

---

## Data Cleaning

The cleaning workflow supports:

* duplicate-row removal;
* constant-column removal;
* median imputation for numerical missing values;
* mean imputation for numerical missing values;
* cleaned dataset preview;
* cleaned CSV download.

Identifier-like columns can be excluded from machine-learning features to reduce the risk of meaningless high-cardinality identifiers influencing model predictions.

---

# Class-Imbalance Analysis

Fraud and churn datasets commonly contain significantly fewer positive cases than negative cases.

The application therefore provides two imbalance-handling strategies.

### Class Weighting

Class weighting increases the importance of minority-class observations during model training without creating additional records.

### SMOTE

SMOTE generates synthetic minority-class training examples to improve representation of the minority class.

Users can select the imbalance strategy before training the models.

---

# Machine Learning

The application trains and compares multiple binary-classification pipelines.

## Classification Models

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier

---

## Classification Metrics

Models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix

The model leaderboard provides a single interface for comparing the candidate models.

For imbalanced risk problems, the application considers metrics beyond accuracy because correctly identifying the minority risk class may be particularly important.

---

# Model Evaluation

The application provides several evaluation views.

### Model Leaderboard

Compares all trained classifiers using the supported evaluation metrics.

### Confusion Matrix

Shows:

* true negatives;
* false positives;
* false negatives;
* true positives.

### ROC Curve

Visualises the relationship between the true-positive rate and false-positive rate across classification thresholds.

### Precision–Recall Curve

Visualises the trade-off between precision and recall.

This is especially useful when the positive risk class is relatively uncommon.

---

# Risk Scoring

After model selection, the application generates row-level risk probabilities.

Example:

```text
Customer
   ↓
Trained Classifier
   ↓
Risk Probability
   ↓
Risk Threshold
   ↓
Risk Flag
   ↓
Risk Band
```

Each row receives:

* risk probability;
* risk flag;
* risk band.

The risk threshold can be adjusted directly from the Streamlit interface.

---

## Risk Bands

Predictions are grouped into interpretable risk categories:

```text
Low
Moderate
High
Critical
```

This makes the probability output easier to inspect from a risk-analysis perspective.

---

# Flagged-Case Analysis

Cases exceeding the selected risk threshold are collected into a flagged-case view.

Users can inspect:

* row ID;
* predicted risk probability;
* risk band;
* risk status.

The application also supports downloading the complete risk report and flagged-case dataset.

---

# Model Explainability

The project includes both global and row-level explainability.

## Global Explainability

Permutation feature importance estimates how strongly model performance depends on individual input features.

The application provides:

* ranked feature-importance table;
* interactive Plotly importance chart;
* strongest overall risk factors.

Permutation importance measures model reliance on a feature and should not be interpreted as evidence of causality.

---

## Individual Risk Explanation

Users can select a particular case and investigate:

```text
Why was this case flagged?
```

The workflow combines the trained model with the selected row to provide a more interpretable view of the individual prediction.

This allows the application to move beyond simply producing a probability score and toward explaining individual risk decisions.

---

# Downloads

The application supports downloading:

* cleaned dataset;
* model leaderboard;
* full risk report;
* flagged cases.

This allows analytical outputs to be used outside the Streamlit interface.

---

# AI Risk Analyst

The application includes an optional LangChain agent powered by the OpenAI API.

The AI layer is designed as an **investigation interface**, rather than replacing the deterministic machine-learning workflow.

Example questions include:

```text
How is the best model performing?
```

```text
What are the strongest risk factors?
```

```text
Which cases have the highest predicted risk?
```

```text
Why is this case considered high risk?
```

The agent uses structured analysis tools to retrieve relevant information before generating an explanation.

---

# Application Architecture

```text
CSV Upload
    │
    ▼
Dataset Validation
    │
    ▼
Working DataFrame
    │
    ├──────────────► Dataset Overview
    │
    ├──────────────► Exploratory Data Analysis
    │
    ├──────────────► Data Cleaning
    │
    └──────────────► Target / Class Balance Analysis
                           │
                           ▼
                  Feature Preprocessing
                           │
                 ┌─────────┴─────────┐
                 │                   │
          Class Weighting          SMOTE
                 │                   │
                 └─────────┬─────────┘
                           ▼
                 Classification Models
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
 Logistic Regression  Decision Tree    Random Forest
                                             │
                                   Gradient Boosting
                           │
                           ▼
                    Model Evaluation
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     Leaderboard     Confusion Matrix   ROC / PR Curves
          │
          ▼
                    Best Model
          │
          ▼
                    Risk Scoring
          │
        ┌─┴──────────────────────┐
        │                        │
Global Explainability    Individual Explanation
        │                        │
        └────────────┬───────────┘
                     ▼
              Structured Results
                     │
                     ▼
               LangChain Agent
                     │
                     ▼
          Natural-Language Analysis
```

---

# Agent Workflow

```text
User Question
    ↓
Current Analysis Context
    ↓
LangChain Agent
    ↓
Agent examines available tools
    ↓
Appropriate tool selected
    ↓
Python tool reads verified analysis results
    ↓
Structured result returned
    ↓
LLM interprets the result
    ↓
Natural-language explanation
```

The language model is not responsible for calculating model metrics or training classifiers.

Those operations remain inside deterministic Python functions.

The agent acts as an interpretation and investigation layer over the resulting analytical outputs.

---

# Machine-Learning Workflow

```text
Selected Binary Target
    ↓
Feature / Target Separation
    ↓
Identifier-Column Detection
    ↓
Numerical / Categorical Detection
    ↓
Train / Test Split
    ↓
Pipeline-Based Preprocessing
    ↓
Class-Imbalance Strategy
    │
    ├── Class Weighting
    │
    └── SMOTE
    ↓
Multiple Classifier Training
    ↓
Model Evaluation
    ↓
Leaderboard Comparison
    ↓
Best-Model Selection
    ↓
Risk Probability Generation
    ↓
Risk Threshold
    ↓
Flagged Cases
    ↓
Explainability
    ↓
Downloads / Agent Investigation
```

---

# Preprocessing Pipeline

Numerical features are processed using a pipeline similar to:

```text
Missing Values
    ↓
Median / Mean Imputation
    ↓
Numerical Transformation
    ↓
Classifier
```

Categorical features are processed using:

```text
Missing Values
    ↓
Categorical Imputation
    ↓
One-Hot Encoding
    ↓
Classifier
```

Preprocessing is kept inside the machine-learning pipeline so the same fitted transformations can be used consistently during model training and inference.

---

# Important Engineering Decisions

## Deterministic Logic and Agent Tools Are Separated

The Streamlit application directly calls complete Python functions for:

* validation;
* cleaning;
* preprocessing;
* model training;
* model evaluation;
* risk scoring;
* explainability.

The AI agent receives smaller structured outputs through controlled tools.

This prevents large model objects and DataFrames from being unnecessarily passed to the language model.

---

## Identifier Columns Are Excluded From Training

Columns such as:

```text
customer_id
transaction_id
account_id
```

may uniquely identify records without containing meaningful predictive information.

The application can detect identifier-like columns and exclude them from model features while preserving them in reports.

---

## Class Imbalance Is Explicitly Handled

Instead of assuming equal target distribution, the application allows users to choose between:

```text
Class Weighting
```

and:

```text
SMOTE
```

This makes imbalance handling an explicit part of the modeling workflow.

---

## Risk Threshold Is User Controlled

A classification probability does not automatically determine how aggressively cases should be flagged.

The application therefore exposes the risk threshold in the interface.

Changing the threshold changes which predictions are treated as flagged cases.

---

## Explainability Is Separate From Prediction

Prediction answers:

```text
How likely is this case to belong to the risk class?
```

Explainability addresses:

```text
Which features is the model relying on?
```

The application presents these as separate analytical stages.

---

# Technology Stack

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* scikit-learn
* imbalanced-learn
* LangChain
* OpenAI API
* python-dotenv

---

# Project Structure

```text
agentic-fraud-churn-risk-analyst/
│
├── app.py
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── .env.example
├── .gitignore
│
├── sample_data/
│   └── risk_customers.csv
│
├── src/
│   │
│   ├── agent/
│   │   ├── agent.py
│   │   ├── context.py
│   │   └── tools.py
│   │
│   ├── analysis/
│   │   └── eda.py
│   │
│   ├── data/
│   │   ├── cleaner.py
│   │   └── validator.py
│   │
│   ├── ml/
│   │   ├── preprocessing.py
│   │   ├── trainer.py
│   │   ├── explainability.py
│   │   └── feature_selection.py
│   │
│   ├── reports/
│   │   └── risk_report.py
│   │
│   └── utils/
│       └── export.py
│
└── tests/
    ├── conftest.py
    └── test_core.py
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/mamathabyreddy99/agentic-fraud-churn-risk-analyst.git
```

```bash
cd agentic-fraud-churn-risk-analyst
```

---

## Create a Virtual Environment

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Configure OpenAI API

The deterministic machine-learning workflow works without an OpenAI API key.

The AI investigation feature requires an OpenAI API key.

Copy:

```text
.env.example
```

to:

```text
.env
```

Then configure the environment variable locally.

```text
OPENAI_API_KEY=your_api_key
```

Do not commit the `.env` file or API credentials to source control.

---

# Start the Application

```bash
streamlit run app.py
```

Then open the local Streamlit application in your browser.

---

# Example Workflow

```text
Upload CSV
    ↓
Select target
    ↓
Inspect dataset
    ↓
Review EDA
    ↓
Clean data
    ↓
Inspect target imbalance
    ↓
Choose Class Weighting or SMOTE
    ↓
Set risk threshold
    ↓
Train models
    ↓
Compare leaderboard
    ↓
Inspect ROC / Precision-Recall curves
    ↓
Generate risk scores
    ↓
Review flagged cases
    ↓
Inspect feature importance
    ↓
Explain individual case
    ↓
Ask AI agent questions
    ↓
Download results
```

---

# Example AI Analyst Questions

```text
How is the selected model performing?
```

```text
What are the strongest global risk factors?
```

```text
Which cases have the highest predicted risk?
```

```text
Explain why a selected case received a high risk score.
```

```text
Summarise the current risk-analysis results.
```

---

# Testing

The repository contains automated tests for core project functionality.

Run:

```bash
pytest
```

The test suite covers core data and machine-learning functionality.

---

# Limitations

* The current workflow focuses on binary classification.
* Model evaluation currently relies on a train/test split rather than full cross-validation.
* Hyperparameter optimisation is limited.
* Risk thresholds require domain-specific validation before production use.
* SMOTE creates synthetic training observations and should be evaluated carefully for each dataset.
* Permutation importance measures model reliance rather than causality.
* Very large datasets may require additional sampling or distributed processing.
* The application is a portfolio/analysis system and would require additional governance, monitoring, authentication, security, and validation controls before production use.
* Predictions should not be treated as proof of fraud, misconduct, or future customer behaviour.

---

# Future Improvements

* Cross-validation
* Hyperparameter optimisation
* SHAP explanations
* Automated threshold optimisation
* Cost-sensitive model evaluation
* Model persistence
* Experiment tracking
* Model drift monitoring
* Automated analytical reports
* Authentication
* Database integration
* API deployment
* Cloud deployment
* Additional agent tools

---

# Security

* API credentials are stored outside source control.
* `.env` is excluded through `.gitignore`.
* Virtual environments are excluded from the repository.
* Private customer or transaction datasets should not be committed.
* Risk outputs should be reviewed before use in real decision-making workflows.

---

# Responsible Use

This application produces machine-learning risk estimates for analytical and decision-support purposes.

A high predicted probability should not be interpreted as proof that a customer committed fraud, will churn, or engaged in misconduct.

Production risk systems require appropriate validation, human review, fairness analysis, security controls, monitoring, and domain-specific governance.

---

# Author

**Mamatha Byreddy**

Interested in applied AI, machine learning, agentic systems, data engineering, and software development.

**GitHub:** `mamathabyreddy99`

**LinkedIn:** https://www.linkedin.com/in/byreddy-mamatha-296a221a8

---

# License 

This project is licensed under the MIT License.
