from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report
from prometheus_client import Gauge
import pandas as pd
from prefect import flow, task
import os
from prometheus_client import start_http_server


start_http_server(8000) 


DRIFT_GAUGE = Gauge(
    "churn_data_drift_detected",
    "1 if drift detected, 0 otherwise"
)

@task(name="load-data")
@task(name="load-data")
def load_data():
    base = "C:/phase 5 pipeline MLOps/phase2_model"
    ref = pd.read_parquet(f"{base}/churn_train_v1.parquet")
    cur = pd.read_parquet(f"{base}/churn_test_v1.parquet")
    return ref, cur

@task(name="align-data")
def align_data(reference, current):
    common_cols = list(set(reference.columns) & set(current.columns))
    reference = reference[common_cols].copy()
    current = current[common_cols].copy()

    null_cols = [
        c for c in common_cols
        if reference[c].isna().all() or current[c].isna().all()
    ]
    if null_cols:
        print(f"Dropping all-null columns: {null_cols}")
    reference = reference.drop(columns=null_cols)
    current = current.drop(columns=null_cols)

    for col in reference.columns:
        if reference[col].dtype != current[col].dtype:
            print(f"Type mismatch on '{col}' — casting both to string")
            reference[col] = reference[col].astype(str)
            current[col] = current[col].astype(str)

    print(f"Aligned on {len(reference.columns)} columns")
    return reference, current

@task(name="detect-drift")
def detect_drift(reference, current):
    # Evidently 0.7.x API
    ref_dataset = Dataset.from_pandas(reference)
    cur_dataset = Dataset.from_pandas(current)

    report = Report([DataDriftPreset()])
    result = report.run(ref_dataset, cur_dataset)

    os.makedirs("outputs", exist_ok=True)
    result.save_html("outputs/drift_report.html")

    result_dict = result.dict()

    drift_detected = False
    for metric in result_dict.get("metrics", []):
        metric_result = metric.get("result", {})
        if "dataset_drift" in metric_result:
            drift_detected = metric_result["dataset_drift"]
            break

    DRIFT_GAUGE.set(1 if drift_detected else 0)

    if drift_detected:
        print("DRIFT DETECTED — retraining needed")
    else:
        print("No drift detected")
    return drift_detected

@flow(name="monitoring-pipeline")
def monitoring_flow():
    ref, cur = load_data()
    ref, cur = align_data(ref, cur)
    drift = detect_drift(ref, cur)
    return drift

if __name__ == "__main__":
    monitoring_flow()