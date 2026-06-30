"""
simulate_new_data.py — Phase 8.1
Appends synthetic rows to the training parquet file to simulate new data arriving.
Run this script to simulate a data ingestion event before triggering retraining.

Usage:
    python simulate_new_data.py --rows 10000
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

BASE = "C:/phase 5 pipeline MLOps/phase2_model"
TRAIN_PATH = f"{BASE}/churn_train_v1.parquet"


def simulate_new_data(n_rows: int = 10000) -> None:
    # Load existing data to match schema and value ranges
    existing = pd.read_parquet(TRAIN_PATH)
    print(f"Existing training rows: {len(existing)}")

    np.random.seed(None)  # different seed each run for variety

    # Generate synthetic rows matching existing schema
    new_data = pd.DataFrame({
        "customer_id": np.random.randint(900000000, 999999999, n_rows),
        "telecom_partner":     np.random.choice(
                                   existing["telecom_partner"].dropna().unique(), n_rows),
        "gender":              np.random.choice(["Male", "Female"], n_rows),
        "age":                 np.random.randint(18, 75, n_rows),
        "state":               np.random.choice(
                                   existing["state"].dropna().unique(), n_rows),
        "city":                np.random.choice(
                                   existing["city"].dropna().unique(), n_rows),
        "pincode":             np.random.randint(100000, 999999, n_rows),
        "date_of_registration": pd.to_datetime(
                                   np.random.choice(
                                       pd.date_range("2018-01-01", "2023-12-31"), n_rows)),
        "num_dependents":      np.random.randint(0, 6, n_rows),
        "estimated_salary":    np.random.randint(20000, 150000, n_rows),
        "calls_made":          np.random.randint(0, 300, n_rows),
        "sms_sent":            np.random.randint(0, 500, n_rows),
        "data_used":           np.random.uniform(0, 50, n_rows).round(2),
        "tenure_days":         np.random.randint(30, 2500, n_rows),
    })

    # Rebuild churn label using same business logic as pipeline_steps.py
    def _norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    logit = (
        3.5
        - 4.0 * _norm(new_data["calls_made"])
        - 3.0 * _norm(new_data["sms_sent"])
        - 2.0 * _norm(new_data["data_used"])
        - 1.5 * _norm(new_data["estimated_salary"])
        + 0.5 * _norm(new_data["num_dependents"])
    )
    prob = 1 / (1 + np.exp(-logit))
    new_data["churn"] = (np.random.rand(n_rows) < prob).astype("Int64")

    # Cast date_of_registration to match existing dtype
    if existing["date_of_registration"].dtype == object:
        new_data["date_of_registration"] = new_data["date_of_registration"].astype(str)

    # Append to existing training data
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined.to_parquet(TRAIN_PATH, index=False)

    print(f"Appended {n_rows} synthetic rows")
    print(f"New training size: {len(combined)} rows")
    print(f"Growth: +{(len(combined) - len(existing)) / len(existing) * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000,
                        help="Number of synthetic rows to append (default: 10000)")
    args = parser.parse_args()
    simulate_new_data(args.rows)
