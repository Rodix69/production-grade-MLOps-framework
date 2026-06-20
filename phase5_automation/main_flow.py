from prefect import flow
from prefect.cache_policies import NO_CACHE
from monitoring_flow import monitoring_flow
from pipeline_steps import (
    ingest_data,
    create_features,
    train_model,
    validate_model,
    register_model,
)
from retrain_manager import (
    check_data_volume_trigger,
    update_row_count_after_training,
    save_version_record,
    get_current_live_version,
)


def run_training_pipeline(trigger: str = "scheduled"):
    """Shared training pipeline used by all retraining triggers."""
    train, val, test                  = ingest_data()
    X_tr, y_tr, X_v, y_v, X_te, y_te = create_features(train, val, test)
    model, run_id                     = train_model(X_tr, y_tr)
    passed, test_auc                  = validate_model(model, run_id, X_v, y_v, X_te, y_te)
    model_version                     = register_model(run_id, passed, test_auc)

    if passed and model_version:
        # 8.4 — record which version is now live
        save_version_record(run_id, model_version, test_auc, trigger=trigger)
        # 8.1 — update row count baseline after successful training
        update_row_count_after_training()

    return passed, test_auc


@flow(name="main-pipeline")
def main_flow():
    print("=== Starting main pipeline ===")

    # Show currently live model before retraining
    print("\n--- Current live model ---")
    get_current_live_version()
    print("--------------------------\n")

    # 8.2 — Run scheduled retraining pipeline
    passed, test_auc = run_training_pipeline(trigger="scheduled")

    # 8.1 — Drift-based retraining trigger
    drift = monitoring_flow()
    if drift:
        print("Drift detected — rerunning training pipeline")
        run_training_pipeline(trigger="drift")

    # 8.1 — Data volume trigger
    should_retrain, current_rows, last_rows = check_data_volume_trigger()
    if should_retrain:
        print(f"Volume trigger — rerunning training pipeline")
        run_training_pipeline(trigger="volume")

    print("=== Pipeline complete ===")


if __name__ == "__main__":
    main_flow()