# 6_evaluate_model.py
# Model Evaluation & Validation Script
# Loads registered model, runs on test data, logs metrics to MLflow

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("   TELECOM CHURN — MODEL EVALUATION SCRIPT")
print("=" * 60)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TRACKING_URI   = "file:./mlruns"
EXPERIMENT     = "telecom-churn-phase4"
MODEL_NAME     = "telecom-churn-xgb"
MODEL_STAGE    = "Production"
TEST_DATA_PATH = "data/churn_test_v1.parquet"
TARGET_COL     = "churn_label"
AUC_THRESHOLD  = 0.80       # minimum AUC to promote
F1_THRESHOLD   = 0.70       # minimum F1  to promote

# ─────────────────────────────────────────────
# STEP 1 — Connect to MLflow
# ─────────────────────────────────────────────
print("\n[STEP 1] Connecting to MLflow...")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()
print(f"✅ Connected — Tracking URI : {TRACKING_URI}")
print(f"✅ Experiment              : {EXPERIMENT}")

# ─────────────────────────────────────────────
# STEP 2 — Load Model from Registry
# ─────────────────────────────────────────────
print(f"\n[STEP 2] Loading model from registry...")
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
model     = mlflow.pyfunc.load_model(model_uri)
print(f"✅ Model loaded — {model_uri}")

# Get model version info
versions = client.get_latest_versions(
    name=MODEL_NAME,
    stages=[MODEL_STAGE]
)
model_version = versions[0].version
print(f"✅ Model version : {model_version}")

# ─────────────────────────────────────────────
# STEP 3 — Load Test Data
# ─────────────────────────────────────────────
print(f"\n[STEP 3] Loading test data...")

if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(
        f"❌ Test data not found at {TEST_DATA_PATH}\n"
        f"   Please run Step 1 in your notebook first."
    )

df_test  = pd.read_parquet(TEST_DATA_PATH)
X_test   = df_test.drop(columns=[TARGET_COL])
y_test   = df_test[TARGET_COL]

print(f"✅ Test data loaded")
print(f"   Shape    : {df_test.shape}")
print(f"   Churners : {y_test.sum()} ({y_test.mean()*100:.1f}%)")

# ─────────────────────────────────────────────
# STEP 4 — Run Predictions
# ─────────────────────────────────────────────
print(f"\n[STEP 4] Running predictions...")
y_pred  = model.predict(X_test)
y_probs = y_pred  # if model returns probabilities

# If model returns raw probabilities, threshold at 0.5
if y_pred.dtype == float:
    y_probs = y_pred
    y_pred  = (y_probs >= 0.5).astype(int)

print(f"✅ Predictions complete")
print(f"   Predicted Churners     : {y_pred.sum()}")
print(f"   Predicted Non-Churners : {len(y_pred) - y_pred.sum()}")

# ─────────────────────────────────────────────
# STEP 5 — Calculate Metrics
# ─────────────────────────────────────────────
print(f"\n[STEP 5] Calculating metrics...")

auc       = roc_auc_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
accuracy  = accuracy_score(y_test, y_pred)

print(f"\n{'─'*40}")
print(f"  AUC       : {auc:.4f}")
print(f"  F1 Score  : {f1:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  Accuracy  : {accuracy:.4f}")
print(f"{'─'*40}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Not Churn", "Churn"]))

# ─────────────────────────────────────────────
# STEP 6 — Promotion Decision
# ─────────────────────────────────────────────
print(f"\n[STEP 6] Checking promotion thresholds...")
print(f"   AUC threshold : {AUC_THRESHOLD} | Got : {auc:.4f}")
print(f"   F1  threshold : {F1_THRESHOLD}  | Got : {f1:.4f}")

promote = auc >= AUC_THRESHOLD and f1 >= F1_THRESHOLD

if promote:
    promotion_decision = "PROMOTE"
    print(f"\n✅ Model PASSED validation gates!")
else:
    promotion_decision = "REJECT"
    print(f"\n❌ Model FAILED validation gates!")

# ─────────────────────────────────────────────
# STEP 7 — Log Everything to MLflow
# ─────────────────────────────────────────────
print(f"\n[STEP 7] Logging results to MLflow...")

with mlflow.start_run(run_name="evaluation_v1"):

    # Log metrics
    mlflow.log_metric("eval_auc",       auc)
    mlflow.log_metric("eval_f1",        f1)
    mlflow.log_metric("eval_precision", precision)
    mlflow.log_metric("eval_recall",    recall)
    mlflow.log_metric("eval_accuracy",  accuracy)

    # Log thresholds
    mlflow.log_param("auc_threshold",   AUC_THRESHOLD)
    mlflow.log_param("f1_threshold",    F1_THRESHOLD)
    mlflow.log_param("model_version",   model_version)
    mlflow.log_param("model_stage",     MODEL_STAGE)
    mlflow.log_param("test_data",       TEST_DATA_PATH)
    mlflow.log_param("test_rows",       len(df_test))

    # Log decision
    mlflow.set_tag("promotion_decision", promotion_decision)
    mlflow.set_tag("phase",              "5 - evaluation")
    mlflow.set_tag("model_name",         MODEL_NAME)

print(f"✅ Results logged to MLflow")

# ─────────────────────────────────────────────
# STEP 8 — Auto Promote or Reject
# ─────────────────────────────────────────────
print(f"\n[STEP 8] Executing promotion decision...")

if promote:
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage="Production"
    )
    print(f"✅ Version {model_version} PROMOTED to Production!")
else:
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage="Archived"
    )
    print(f"❌ Version {model_version} ARCHIVED — did not meet thresholds")

print("\n" + "=" * 60)
print(f"   EVALUATION COMPLETE — Decision : {promotion_decision}")
print("=" * 60)