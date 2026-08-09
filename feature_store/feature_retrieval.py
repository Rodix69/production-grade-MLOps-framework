"""
Phase 3 — Feature Retrieval
Shows how to retrieve features from the Feast feature store
for both training and online inference.

Usage:
    python feature_retrieval.py
"""

import pandas as pd
from datetime import datetime, timezone
from feast import FeatureStore


# All 27 model features grouped by feature view
ALL_FEATURES = [
    # Demographics
    "customer_demographics:telecom_partner",
    "customer_demographics:gender",
    "customer_demographics:age",
    "customer_demographics:state",
    "customer_demographics:city",
    "customer_demographics:pincode",
    "customer_demographics:num_dependents",
    "customer_demographics:estimated_salary",
    # Usage
    "customer_usage:calls_made",
    "customer_usage:sms_sent",
    "customer_usage:data_used",
    "customer_usage:tenure_days",
    # Engineered
    "customer_engineered:total_activity",
    "customer_engineered:avg_data_per_day",
    "customer_engineered:avg_calls_per_day",
    "customer_engineered:avg_sms_per_day",
    "customer_engineered:engagement_score",
    "customer_engineered:low_activity_flag",
    "customer_engineered:high_value_user",
    "customer_engineered:partner_med_calls",
    "customer_engineered:partner_med_data",
    "customer_engineered:partner_med_sms",
    "customer_engineered:calls_vs_partner",
    "customer_engineered:data_vs_partner",
    "customer_engineered:is_new_customer",
    "customer_engineered:calls_intensity",
    "customer_engineered:data_intensity",
]


def get_training_features(customer_ids: list) -> pd.DataFrame:
    """
    Retrieve historical features for model training.
    Used in Phase 4 model training pipeline.
    """
    store = FeatureStore(repo_path=".")

    entity_df = pd.DataFrame({
        "customer_id":     customer_ids,
        "event_timestamp": [pd.Timestamp("2024-01-01", tz="UTC")] * len(customer_ids),
    })

    features_df = store.get_historical_features(
        entity_df=entity_df,
        features=ALL_FEATURES,
    ).to_df()

    print(f"Retrieved {len(features_df)} rows with {len(features_df.columns)} columns")
    return features_df


def get_online_features(customer_ids: list) -> pd.DataFrame:
    """
    Retrieve online features for real-time inference.
    Used by the FastAPI /predict endpoint in Phase 6.
    """
    store = FeatureStore(repo_path=".")

    feature_vector = store.get_online_features(
        features=ALL_FEATURES,
        entity_rows=[{"customer_id": cid} for cid in customer_ids],
    ).to_df()

    print(f"Online features retrieved for {len(customer_ids)} customers")
    return feature_vector


if __name__ == "__main__":
    store = FeatureStore(repo_path=".")

    print("=" * 60)
    print("Phase 3 — Feature Retrieval Demo")
    print("=" * 60)

    # Historical features (for training)
    print("\n1. Historical Feature Retrieval (for training):")
    train_features = get_training_features(customer_ids=[0, 1, 2, 3, 4])
    print(train_features[["customer_id", "engagement_score", "calls_intensity", "tenure_days"]].to_string(index=False))

    # Online features (for inference)
    print("\n2. Online Feature Retrieval (for inference):")
    online_features = get_online_features(customer_ids=[0, 1, 2])
    print(online_features[["customer_id", "engagement_score", "calls_intensity"]].to_string(index=False))

    print("\nFeature retrieval demo complete.")
