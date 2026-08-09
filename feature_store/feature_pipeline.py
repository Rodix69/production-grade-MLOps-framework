"""
Phase 3 - Feature Pipeline
Materializes engineered features into the Feast feature store.

Usage:
    python feature_pipeline.py
"""

import os
import pandas as pd
from datetime import datetime, timezone
from feast import FeatureStore


# ── Step 1: Validate parquet files ────────────────────────────────────────────
def validate_data():
    print("Step 1: Validating feature data...")
    for split in ["train", "val", "test"]:
        path = f"data/churn_{split}_v1.parquet"
        df = pd.read_parquet(path)
        nulls = df.isnull().sum().sum()
        print(f"  {split}: {df.shape[0]:,} rows | {df.shape[1]} cols | {nulls} nulls")
    print("  Validation passed.")


# ── Step 2: Apply Feast definitions ──────────────────────────────────────────
def apply_feast():
    print("\nStep 2: Applying Feast feature definitions...")
    os.system("feast apply")
    print("  Feast apply complete.")


# ── Step 3: Materialize features ─────────────────────────────────────────────
def materialize():
    print("\nStep 3: Materializing features into online store...")
    store = FeatureStore(repo_path=".")
    store.materialize(
        start_date=pd.Timestamp("2025-01-01", tz="UTC"),
        end_date=pd.Timestamp("2026-05-25", tz="UTC"),
    )
    print("  Materialization complete.")


# ── Step 4: Validate historical retrieval ────────────────────────────────────
def validate_historical():
    print("\nStep 4: Validating historical feature retrieval...")
    store = FeatureStore(repo_path=".")

    entity_df = pd.DataFrame({
        "customer_id":     [0, 1, 2, 3, 4],
        "event_timestamp": [pd.Timestamp("2025-06-01", tz="UTC")] * 5,
    })

    features = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "customer_engineered:engagement_score",
            "customer_engineered:calls_intensity",
            "customer_engineered:data_intensity",
            "customer_demographics:age",
            "customer_demographics:estimated_salary",
            "customer_usage:tenure_days",
            "customer_usage:calls_made",
        ],
    ).to_df()

    print("  Historical features retrieved successfully:")
    print(features.to_string(index=False))
    return features


# ── Step 5: Validate online retrieval ────────────────────────────────────────
def validate_online():
    print("\nStep 5: Validating online feature retrieval...")
    store = FeatureStore(repo_path=".")

    features = store.get_online_features(
        features=[
            "customer_engineered:engagement_score",
            "customer_engineered:calls_intensity",
            "customer_usage:tenure_days",
            "customer_demographics:age",
        ],
        entity_rows=[{"customer_id": i} for i in range(3)],
    ).to_df()

    print("  Online features retrieved successfully:")
    print(features.to_string(index=False))
    return features


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 - Telecom Churn Feature Pipeline")
    print("=" * 60)

    validate_data()
    apply_feast()
    materialize()
    validate_historical()
    validate_online()

    print("\n" + "=" * 60)
    print("Feature pipeline complete!")
    print("=" * 60)