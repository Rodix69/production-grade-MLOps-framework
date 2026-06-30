import mlflow
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = mlflow.tracking.MlflowClient()

versions = client.search_model_versions("name='telecom_churn_champion'")

# Find the version with the best AUC
best_version = None
best_auc     = -1

for v in versions:
    try:
        run_info = client.get_run(v.run_id)
        auc      = run_info.data.metrics.get("test_auc", 0)
        if auc > best_auc:
            best_auc     = auc
            best_version = v.version
    except Exception:
        continue

print(f"Best version: v{best_version} with AUC {best_auc:.4f} — keeping in Production")

# Archive everything else in Production
for v in versions:
    if v.current_stage == "Production" and v.version != best_version:
        client.transition_model_version_stage(
            name="telecom_churn_champion",
            version=v.version,
            stage="Archived"
        )
        print(f"Archived v{v.version}")

# Make sure best version is in Production
client.transition_model_version_stage(
    name="telecom_churn_champion",
    version=best_version,
    stage="Production"
)
print(f"v{best_version} confirmed in Production")