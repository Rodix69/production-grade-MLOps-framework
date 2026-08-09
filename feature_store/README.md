# Phase 3: Feature Store — Telecom Churn

## Overview

This phase implements a **Feast feature store** for the telecom churn prediction project. It stores, versions, and serves the 27 engineered features used by the logistic regression model deployed in Phase 6.

---

## Project Structure

```
feature_store/
├── feature_store.yaml        ← Feast configuration (local provider, SQLite)
├── feature_definitions.py    ← Entity, FeatureView, and Feature definitions
├── feature_pipeline.py       ← Feature engineering + materialization pipeline
├── feature_retrieval.py      ← Historical and online feature retrieval
├── README.md                 ← This file
└── data/
    ├── churn_train_v1.parquet   ← Training features (170,487 rows, 30 cols)
    ├── churn_val_v1.parquet     ← Validation features (36,533 rows, 30 cols)
    ├── churn_test_v1.parquet    ← Test features (36,533 rows, 30 cols)
    ├── registry.db              ← Feast feature registry (auto-generated)
    └── online_store.db          ← Feast online store SQLite (auto-generated)
```

---

## Feature Views

### 1. customer_demographics
Raw demographic features about the customer.

| Feature | Type | Description |
|---------|------|-------------|
| telecom_partner | String | Mobile network provider |
| gender | String | Customer gender |
| age | Int64 | Customer age |
| state | String | State of residence |
| city | String | City of residence |
| pincode | Int64 | PIN code |
| num_dependents | Int64 | Number of dependents |
| estimated_salary | Float64 | Estimated annual salary |

### 2. customer_usage
Raw usage behavior features.

| Feature | Type | Description |
|---------|------|-------------|
| calls_made | Int64 | Total calls made |
| sms_sent | Int64 | Total SMS sent |
| data_used | Float64 | Total data used (GB) |
| tenure_days | Int64 | Days as a customer |

### 3. customer_engineered
Engineered features created in Phase 3 feature engineering.

| Feature | Type | Description |
|---------|------|-------------|
| total_activity | Float64 | Sum of calls + sms + data |
| avg_data_per_day | Float64 | Daily average data usage |
| avg_calls_per_day | Float64 | Daily average calls |
| avg_sms_per_day | Float64 | Daily average SMS |
| engagement_score | Float64 | Weighted engagement metric |
| low_activity_flag | Int64 | 1 if below 25th percentile activity |
| high_value_user | Int64 | 1 if above 75th percentile salary |
| partner_med_calls | Float64 | Median calls for telecom partner |
| partner_med_data | Float64 | Median data for telecom partner |
| partner_med_sms | Float64 | Median SMS for telecom partner |
| calls_vs_partner | Float64 | Customer calls relative to partner median |
| data_vs_partner | Float64 | Customer data relative to partner median |
| is_new_customer | Int64 | 1 if tenure below 25th percentile |
| calls_intensity | Float64 | Calls relative to dataset median |
| data_intensity | Float64 | Data relative to dataset median |

---

## Setup and Usage

### 1. Install Feast
```bash
pip install feast
```

### 2. Apply feature definitions
```bash
cd feature_store
feast apply
```

### 3. Run feature pipeline
```bash
python feature_pipeline.py
```

### 4. Retrieve features
```bash
python feature_retrieval.py
```

---

## How It Links to Other Phases

| Phase | How Feature Store is Used |
|-------|--------------------------|
| Phase 3 (this) | Features defined, engineered, and stored |
| Phase 4 | Training features retrieved from store for model training |
| Phase 6 | Online features retrieved for real-time inference via FastAPI |
| Phase 7 | Feature distributions monitored for drift detection |

---

## Deliverables

- [x] Feature store implementation (Feast, local provider)
- [x] Feature definitions (3 feature views, 27 features)
- [x] Feature pipeline script (engineering + materialization)
- [x] Feature retrieval script (historical + online)
- [x] Training data stored as versioned parquet files
- [x] Feature versions linked to MLflow runs via tags
