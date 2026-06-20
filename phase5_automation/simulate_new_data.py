"""
simulate_new_data.py — Phase 8.1
Appends synthetic rows to the training parquet file to simulate new data arriving.
Run this script to simulate a data ingestion event before triggering retraining.

A timestamped backup of the original training file is created automatically
before any changes are made (see backups/ folder). Use --restore-latest to
revert to the most recent backup.

Usage:
    python simulate_new_data.py --rows 10000
    python simulate_new_data.py --restore-latest
"""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE        = os.getenv("DATA_DIR", "data/processed")
TRAIN_PATH  = Path(f"{BASE}/churn_train_v1.parquet")
BACKUP_DIR  = Path(f"{BASE}/backups")


def backup_train_file() -> Path:
    """Copy the current training parquet to a timestamped backup before mutating it."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"churn_train_v1_{timestamp}.parquet"
    shutil.copy2(TRAIN_PATH, backup_path)
    print(f"Backup created: {backup_path}")
    return backup_path


def restore_latest_backup() -> None:
    """Restore the training parquet from the most recent backup."""
    if not BACKUP_DIR.exists():
        print("No backups directory found — nothing to restore.")
        return

    backups = sorted(BACKUP_DIR.glob("churn_train_v1_*.parquet"))
    if not backups:
        print("No backups found — nothing to restore.")
        return

    latest = backups[-1]
    shutil.copy2(latest, TRAIN_PATH)
    print(f"Restored {TRAIN_PATH} from {latest}")


def simulate_new_data(n_rows: int = 10000) -> None:
    # Back up the original file before any mutation — this is what was
    # previously a manual, easy-to-forget step.
    backup_train_file()

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
    print(f"If you need to undo this, run: python simulate_new_data.py --restore-latest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000,
                        help="Number of synthetic rows to append (default: 10000)")
    parser.add_argument("--restore-latest", action="store_true",
                        help="Restore churn_train_v1.parquet from the most recent backup")
    args = parser.parse_args()

    if args.restore_latest:
        restore_latest_backup()
    else:
        simulate_new_data(args.rows)