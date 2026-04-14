# 5_feature_pipeline.py
import pandas as pd, numpy as np, joblib, json
from pathlib import Path

def build_features(df: pd.DataFrame, fit: bool, artifacts_dir="models/v1") -> pd.DataFrame:
    """
    fit=True  → compute thresholds from df (training time)
    fit=False → load saved thresholds and apply (serving time)
    """
    df = df.copy()

    # Rate features (no leakage — division only)
    df['total_activity']    = df['calls_made'] + df['sms_sent'] + df['data_used']
    df['avg_calls_per_day'] = df['calls_made'] / (df['tenure_days'] + 1)
    df['avg_data_per_day']  = df['data_used']  / (df['tenure_days'] + 1)
    df['avg_sms_per_day']   = df['sms_sent']   / (df['tenure_days'] + 1)
    df['engagement_score']  = (0.5*df['calls_made'] +
                                0.3*df['sms_sent'] +
                                0.2*df['data_used'])

    if fit:
        # Compute and save thresholds from THIS data (training only)
        thresholds = {
            'activity_thresh': float(df['total_activity'].quantile(0.25)),
            'salary_thresh':   float(df['estimated_salary'].quantile(0.75)),
        }
        joblib.dump(thresholds, f"{artifacts_dir}/feature_thresholds.pkl")
    else:
        # Load pre-computed thresholds (serving / val / test)
        thresholds = joblib.load(f"{artifacts_dir}/feature_thresholds.pkl")

    df['low_activity_flag'] = (df['total_activity'] < thresholds['activity_thresh']).astype(int)
    df['high_value_user']   = (df['estimated_salary'] > thresholds['salary_thresh']).astype(int)

    return df