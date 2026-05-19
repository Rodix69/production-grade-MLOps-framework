# Phase 4: Model Development and Experiment Tracking

## Overview

This phase implements a production-grade machine learning experimentation workflow for telecom churn prediction using MLflow, Logistic Regression, and XGBoost.

The objective of this phase is to:

* Train baseline and candidate models
* Track experiments and metrics
* Compare model performance
* Register models in MLflow Model Registry
* Select a champion production model
* Implement staging and production workflows

---

# Technologies Used

| Component             | Tool                            |
| --------------------- | ------------------------------- |
| Experiment Tracking   | MLflow                          |
| Baseline Model        | Logistic Regression             |
| Candidate Model       | XGBoost                         |
| Hyperparameter Tuning | Manual tuning / MLflow tracking |
| Data Processing       | Pandas, NumPy                   |
| ML Framework          | Scikit-learn, XGBoost           |
| Model Registry        | MLflow Model Registry           |
| Version Control       | Git & GitHub                    |

---

# Project Structure

```text
production-grade-MLOps-framework/
│
├── data/
├── notebooks/
├── src/
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Phase 4 Workflow

## 1. Baseline Model Training

A Logistic Regression model was trained as the baseline model.

### Metrics Logged

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

The baseline model serves as the benchmark for candidate model comparison.

---

## 2. XGBoost Candidate Model

A tuned XGBoost classifier was trained and evaluated against the baseline model.

### Logged Parameters

* learning_rate
* max_depth
* n_estimators
* threshold
* regularization parameters

### Logged Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

# MLflow Experiment Tracking

MLflow was used to:

* Track training runs
* Log metrics and parameters
* Store model artifacts
* Compare experiments
* Register models
* Manage staging and production workflows

## Experiment Name

```text
telecom-churn-phase4
```

---

# Registered Models

| Model                | Purpose                      |
| -------------------- | ---------------------------- |
| telecom-churn-logreg | Production champion model    |
| telecom-churn-xgb    | Candidate / challenger model |

---

# Champion Model Selection

## Experiment Results

| Model               | ROC-AUC |
| ------------------- | ------- |
| Logistic Regression | 0.8302  |
| XGBoost             | 0.8296  |

## Final Decision

The Logistic Regression model was selected as the Production champion model because:

* It achieved slightly higher ROC-AUC
* It has lower complexity
* It is easier to interpret
* It provides faster inference performance

The XGBoost model was retained as a candidate model in the staging workflow.

---

# MLflow Registry Workflow

## Production Model

```text
telecom-churn-logreg
```

## Staging Model

```text
telecom-churn-xgb
```

---

# Key Features Implemented

* MLflow experiment tracking
* Baseline vs challenger workflow
* Model registry integration
* Metrics and parameter logging
* Artifact tracking
* Champion/challenger evaluation
* Git-integrated project structure
* Production-ready experimentation workflow

---

# Deliverables Completed

* Experiment tracking setup
* Model training pipeline
* Baseline model evaluation
* Candidate model evaluation
* MLflow integration
* Model registry workflow
* Champion model selection
* GitHub repository organization

---

# Next Phase

## Phase 5: Pipeline Automation (CI/CD for ML)

Upcoming tasks:

* Automated training pipelines
* CI/CD workflows
* Validation gates
* Automated model registration
* Deployment automation

---

# Author

Sreehari937

Production-Grade MLOps Framework Project
