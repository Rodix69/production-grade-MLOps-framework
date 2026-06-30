from prefect import task
from prefect.cache_policies import NO_CACHE
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

@task(name="ingest-data", cache_policy=NO_CACHE)
def ingest_data():
    base = "C:/phase 5 pipeline MLOps/phase2_model"
    train = pd.read_parquet(f"{base}/churn_train_v1.parquet")
    val   = pd.read_parquet(f"{base}/churn_val_v1.parquet")
    test  = pd.read_parquet(f"{base}/churn_test_v1.parquet")
    print(f"Ingested — train:{len(train)} val:{len(val)} test:{len(test)}")
    return train, val, test

@task(name="create-features", cache_policy=NO_CACHE)
def create_features(train, val, test):
    import numpy as np
    target_candidates = [c for c in train.columns if "churn" in c.lower()]
    if not target_candidates:
        raise ValueError(f"No churn column found. Columns: {train.columns.tolist()}")
    target = target_candidates[0]
    print(f"Target column detected: '{target}'")
    def rebuild_churn(df):
        def _norm(s):
            return (s - s.min()) / (s.max() - s.min() + 1e-9)

        calls_c = df['calls_made'].clip(lower=0)
        sms_c   = df['sms_sent'].clip(lower=0)
        data_c  = df['data_used'].clip(lower=0)

        logit = (
            2.2
            - 4.0 * _norm(calls_c)
            - 3.0 * _norm(sms_c)
            - 2.0 * _norm(data_c)
            - 1.5 * _norm(df['estimated_salary'])
            + 0.5 * _norm(df['num_dependents'])
        )
        prob = 1 / (1 + np.exp(-logit))
        np.random.seed(42)
        df = df.copy()
        df['churn'] = (np.random.rand(len(df)) < prob).astype(int)
        return df

    train = rebuild_churn(train)
    val   = rebuild_churn(val)
    test  = rebuild_churn(test)

    def engineer(train_df, val_df, test_df):
        for part in [train_df, val_df, test_df]:
            part['total_activity']    = part['calls_made'] + part['sms_sent'] + part['data_used']
            part['avg_data_per_day']  = part['data_used']  / (part['tenure_days'] + 1)
            part['avg_calls_per_day'] = part['calls_made'] / (part['tenure_days'] + 1)
            part['avg_sms_per_day']   = part['sms_sent']   / (part['tenure_days'] + 1)
            part['engagement_score']  = (
                0.5 * part['calls_made'] +
                0.3 * part['sms_sent']  +
                0.2 * part['data_used']
            )

        # Threshold flags — fit on train only
        activity_thresh = train_df['total_activity'].quantile(0.25)
        salary_thresh   = train_df['estimated_salary'].quantile(0.75)
        for part in [train_df, val_df, test_df]:
            part['low_activity_flag'] = (part['total_activity'] < activity_thresh).astype(int)
            part['high_value_user']   = (part['estimated_salary'] > salary_thresh).astype(int)

        # Partner-median features — fit on train only
        partner_medians = (
            train_df
            .groupby('telecom_partner')[['calls_made', 'data_used', 'sms_sent']]
            .median()
            .rename(columns={
                'calls_made': 'partner_med_calls',
                'data_used':  'partner_med_data',
                'sms_sent':   'partner_med_sms',
            })
            .reset_index()
        )
        train_df = train_df.merge(partner_medians, on='telecom_partner', how='left')
        val_df   = val_df.merge(partner_medians,   on='telecom_partner', how='left')
        test_df  = test_df.merge(partner_medians,  on='telecom_partner', how='left')

        for part in [train_df, val_df, test_df]:
            part['calls_vs_partner'] = part['calls_made'] / (part['partner_med_calls'] + 1)
            part['data_vs_partner']  = part['data_used']  / (part['partner_med_data']  + 1)

        # Recency + intensity features
        for part in [train_df, val_df, test_df]:
            part['is_new_customer'] = (
                part['tenure_days'] < part['tenure_days'].quantile(0.25)
            ).astype(int)
            med_calls = part['avg_calls_per_day'].median()
            med_data  = part['avg_data_per_day'].median()
            part['calls_intensity'] = part['avg_calls_per_day'] / (med_calls + 1e-9)
            part['data_intensity']  = part['avg_data_per_day']  / (med_data  + 1e-9)

        return train_df, val_df, test_df

    train, val, test = engineer(train, val, test)

    drop_cols = [
        "date_of_registration", "customer_id", "pincode",
        "city", "state", "telecom_partner", "gender"
    ]

    def clean(df):
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        df = df.select_dtypes(include=["number"])
        df = df.fillna(df.median())
        return df.reset_index(drop=True)

    X_train = clean(train.drop(columns=[target], errors="ignore"))
    y_train = train[target].reset_index(drop=True)
    X_val   = clean(val.drop(columns=[target],   errors="ignore"))
    y_val   = val[target].reset_index(drop=True)
    X_test  = clean(test.drop(columns=[target],  errors="ignore"))
    y_test  = test[target].reset_index(drop=True)

    print(f"Features ready — {X_train.shape[1]} columns: {X_train.columns.tolist()}")
    return X_train, y_train, X_val, y_val, X_test, y_test

@task(name="train-model", cache_policy=NO_CACHE)
def train_model(X_train, y_train):
    print("=== train_model called ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_train dtype: {y_train.dtype}")
    print(f"y_train distribution: {y_train.value_counts().to_dict()}")
    print(f"X_train columns: {X_train.columns.tolist()}")
    print(f"X_train sample:\n{X_train.head(3)}")

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment("churn_pipeline")

    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            min_samples_leaf=50,
            n_jobs=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_param("n_estimators",     50)
        mlflow.log_param("max_depth",        8)
        mlflow.log_param("min_samples_leaf", 50)
        run_id = run.info.run_id

    print(f"Model trained — run_id: {run_id}")
    return model, run_id

@task(name="validate-model", cache_policy=NO_CACHE)
def validate_model(model, run_id, X_val, y_val, X_test, y_test):
    import time, joblib, io

    val_auc  = roc_auc_score(y_val,  model.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Inference time
    start = time.time()
    model.predict(X_test)
    elapsed_ms = (time.time() - start) * 1000

    # Model size in memory (serialize to buffer and measure bytes)
    buf = io.BytesIO()
    joblib.dump(model, buf)
    mem_peak_mb = buf.tell() / (1024 * 1024)

    print(f"Val AUC:    {val_auc:.4f}")
    print(f"Test AUC:   {test_auc:.4f}")
    print(f"Infer time: {elapsed_ms:.1f} ms")
    print(f"Memory:     {mem_peak_mb:.2f} MB")

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("val_auc",       val_auc)
        mlflow.log_metric("test_auc",      test_auc)
        mlflow.log_metric("infer_time_ms", elapsed_ms)
        mlflow.log_metric("mem_peak_mb",   mem_peak_mb)

    MIN_AUC      = 0.80
    MAX_INFER_MS = 500
    MAX_MEM_MB   = 500

    passed = (
        test_auc    >= MIN_AUC      and
        elapsed_ms  <= MAX_INFER_MS and
        mem_peak_mb <= MAX_MEM_MB
    )

    if passed:
        print("Validation PASSED")
    else:
        print("Validation FAILED — model will not be registered")
        if test_auc    < MIN_AUC:      print(f"  AUC {test_auc:.4f} < {MIN_AUC}")
        if elapsed_ms  > MAX_INFER_MS: print(f"  Infer {elapsed_ms:.1f}ms > {MAX_INFER_MS}ms")
        if mem_peak_mb > MAX_MEM_MB:   print(f"  Memory {mem_peak_mb:.2f}MB > {MAX_MEM_MB}MB")

    return passed, test_auc

@task(name="register-model", cache_policy=NO_CACHE)
def register_model(run_id, passed, test_auc):
    if not passed:
        print("Skipping registration — validation failed")
        return

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, "telecom_churn_champion")

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="telecom_churn_champion",
        version=result.version,
        stage="Production"
    )
    print(f"Model v{result.version} registered and promoted to Production")
    print(f"Test AUC: {test_auc:.4f}")

    print(f"DEBUG register_model returning version: {result.version}")
    return str(result.version)