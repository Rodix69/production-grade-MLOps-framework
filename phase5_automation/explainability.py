"""
explainability.py — Phase 9.2
Generates SHAP-based model explainability reports.
Produces feature importance plots and individual prediction explanations.

Usage:
    python explainability.py
    from explainability import generate_shap_report, explain_single_prediction
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
OUTPUT_DIR = "C:/phase 5 pipeline MLOps/phase5_automation/outputs/explainability"
BASE       = "C:/phase 5 pipeline MLOps/phase2_model"


def _get_shap_2d(shap_values):
    """
    Always returns a 2D array (n_samples, n_features) for class 1 (churn).
    Handles all SHAP output formats across versions.
    """
    if isinstance(shap_values, list):
        # Old SHAP: list of [class0_array, class1_array]
        sv = np.array(shap_values[1])
    else:
        sv = np.array(shap_values)

    if sv.ndim == 3:
        # Shape (n_samples, n_features, n_classes) — take class 1
        sv = sv[:, :, 1]
    elif sv.ndim == 1:
        sv = sv.reshape(1, -1)

    return sv  # guaranteed 2D: (n_samples, n_features)


def generate_shap_report(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                         run_id: str = None, n_samples: int = 500) -> dict:
    """
    Generate SHAP explainability report for the trained model.
    Saves plots and summary to outputs/explainability/.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return {}

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Generating SHAP report on {n_samples} samples...")

    X_sample    = X_test.sample(min(n_samples, len(X_test)), random_state=42)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Always get clean 2D array
    sv = _get_shap_2d(shap_values)

    # Feature importance
    mean_abs_shap    = np.abs(sv).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature":    X_sample.columns.tolist(),
        "shap_value": mean_abs_shap.tolist(),
    }).sort_values("shap_value", ascending=False).reset_index(drop=True)

    print("\nTop 10 features by SHAP importance:")
    print(feature_importance.head(10).to_string(index=False))

    # Plot 1: Feature importance bar chart
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        feature_importance["feature"][:15][::-1],
        feature_importance["shap_value"][:15][::-1],
        color="#2196F3"
    )
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Feature Importance (SHAP) — Telecom Churn Model", fontsize=14)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    bar_path = f"{OUTPUT_DIR}/shap_feature_importance.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bar_path}")

    # Plot 2: SHAP summary dot plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, show=False, plot_size=None)
    plt.title("SHAP Summary Plot — Telecom Churn Model", fontsize=14)
    plt.tight_layout()
    summary_plot_path = f"{OUTPUT_DIR}/shap_summary_plot.png"
    plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {summary_plot_path}")

    # Plot 3: Dependence plot for top feature
    top_feature = feature_importance["feature"].iloc[0]
    dep_path    = f"{OUTPUT_DIR}/shap_dependence_{top_feature}.png"
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(top_feature, sv, X_sample, show=False, ax=ax)
        plt.title(f"SHAP Dependence — {top_feature}", fontsize=13)
        plt.tight_layout()
        plt.savefig(dep_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {dep_path}")
    except Exception as e:
        print(f"Dependence plot skipped: {e}")
        dep_path = None

    # Save JSON summary
    summary = {
        "run_id":            run_id,
        "generated_at":      pd.Timestamp.now().isoformat(),
        "samples_explained": n_samples,
        "top_features":      feature_importance.head(10).to_dict(orient="records"),
        "plots": {
            "feature_importance": bar_path,
            "summary_plot":       summary_plot_path,
            "dependence_plot":    dep_path or "skipped",
        }
    }
    summary_json_path = f"{OUTPUT_DIR}/shap_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_json_path}")

    # Log to MLflow
    if run_id:
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(bar_path,          "explainability")
                mlflow.log_artifact(summary_plot_path, "explainability")
                mlflow.log_artifact(summary_json_path, "explainability")
                if dep_path:
                    mlflow.log_artifact(dep_path, "explainability")
                mlflow.log_metric("top_feature_shap",
                                  float(feature_importance["shap_value"].iloc[0]))
            print(f"Explainability artifacts logged to MLflow run {run_id}")
        except Exception as e:
            print(f"Warning: Could not log to MLflow: {e}")

    return summary


def explain_single_prediction(model, sample: pd.DataFrame) -> dict:
    """
    Explain a single prediction using SHAP.
    Returns feature contributions for one customer.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return {}

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Get 2D array then take first row
    sv = _get_shap_2d(shap_values)
    sv_row = sv[0]  # 1D array of shape (n_features,)

    cols = sample.columns.tolist()
    contributions = sorted(
        zip(cols, sv_row.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    prediction  = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0][1])

    result = {
        "prediction":        prediction,
        "churn_probability": round(probability, 4),
        "churn_status":      "Churn" if prediction == 1 else "No Churn",
        "top_contributors":  [
            {"feature": f, "shap_value": round(float(v), 4)}
            for f, v in contributions[:10]
        ]
    }

    print(f"\nPrediction: {result['churn_status']} "
          f"(probability: {result['churn_probability']:.1%})")
    print("\nTop feature contributions:")
    for c in result["top_contributors"][:5]:
        direction = "increases churn" if c["shap_value"] > 0 else "decreases churn"
        print(f"  {c['feature']:<25} {c['shap_value']:>8.4f}  {direction}")

    return result


if __name__ == "__main__":
    import sys
    import mlflow

    print("Loading model and data for explainability report...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.get_latest_versions(
            "telecom_churn_champion", stages=["Production"]
        )
        if not versions:
            print("No Production model found in MLflow registry.")
            sys.exit(1)

        model   = mlflow.sklearn.load_model("models:/telecom_churn_champion/Production")
        run_id  = versions[0].run_id
        print(f"Loaded Production model — run_id: {run_id}")

    except Exception as e:
        print(f"Could not load model from MLflow: {e}")
        sys.exit(1)

    # Load data
    train = pd.read_parquet(f"{BASE}/churn_train_v1.parquet")
    val   = pd.read_parquet(f"{BASE}/churn_val_v1.parquet")
    test  = pd.read_parquet(f"{BASE}/churn_test_v1.parquet")

    sys.path.insert(0, "C:/phase 5 pipeline MLOps/phase5_automation")
    from pipeline_steps import create_features
    X_tr, y_tr, X_v, y_v, X_te, y_te = create_features.fn(train, val, test)

    # Generate full report
    summary = generate_shap_report(model, X_tr, X_te, run_id=run_id)

    # Explain one sample
    print("\n--- Single Prediction Explanation ---")
    explain_single_prediction(model, X_te.iloc[[0]])