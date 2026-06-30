"""
approval_workflow.py — Phase 9.3
Implements Dev → Staging → Production approval workflow for model registry.
Models must pass each stage gate before being promoted to Production.

Usage:
    python approval_workflow.py promote --version 3
    python approval_workflow.py status
    python approval_workflow.py history
"""

import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.tracking

MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME    = "telecom_churn_champion"
WORKFLOW_FILE = "C:/phase 5 pipeline MLOps/phase5_automation/approval_workflow.json"

# ─── Stage Gates ──────────────────────────────────────────────────────────────

STAGE_GATES = {
    "Staging": {
        "min_auc":        0.78,   # slightly relaxed for staging
        "max_infer_ms":   500,
        "max_mem_mb":     500,
        "description":    "Initial validation gate — model moves from dev to staging",
    },
    "Production": {
        "min_auc":        0.80,   # stricter for production
        "max_infer_ms":   500,
        "max_mem_mb":     500,
        "description":    "Final production gate — model serves live traffic",
    },
}

# Stage progression
STAGE_ORDER = ["None", "Staging", "Production", "Archived"]


# ─── 9.3 Approval Workflow ────────────────────────────────────────────────────

def promote_model(version: str, target_stage: str,
                  approver: str = "automated_pipeline") -> bool:
    """
    Promote a model version through the approval workflow.
    Checks stage gates before allowing promotion.

    Args:
        version:      Model version number
        target_stage: 'Staging' or 'Production'
        approver:     Who approved (person or system)

    Returns:
        True if promotion succeeded, False if gates failed
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    print(f"\nPromotion request: v{version} → {target_stage}")
    print(f"Approver: {approver}")

    # Fetch model version metrics from MLflow
    try:
        mv       = client.get_model_version(MODEL_NAME, version)
        run_id   = mv.run_id
        run_info = client.get_run(run_id)
        metrics  = run_info.data.metrics
    except Exception as e:
        print(f"Could not fetch model metrics: {e}")
        return False

    test_auc   = metrics.get("test_auc",      0)
    infer_ms   = metrics.get("infer_time_ms", 9999)
    mem_mb     = metrics.get("mem_peak_mb",   9999)

    print(f"\nModel metrics:")
    print(f"  Test AUC   : {test_auc:.4f}")
    print(f"  Infer time : {infer_ms:.1f} ms")
    print(f"  Memory     : {mem_mb:.2f} MB")

    # Check stage gates
    gates  = STAGE_GATES.get(target_stage, {})
    passed = _check_gates(test_auc, infer_ms, mem_mb, gates, target_stage)

    if not passed:
        _record_decision(version, target_stage, "REJECTED", approver,
                        test_auc, infer_ms, mem_mb,
                        reason="Failed stage gate checks")
        return False

    # Archive current production model if promoting to Production
    if target_stage == "Production":
        _archive_current_production(client, version)

    # Promote
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=version, stage=target_stage
        )
        print(f"\n✓ v{version} promoted to {target_stage}")
        _record_decision(version, target_stage, "APPROVED", approver,
                        test_auc, infer_ms, mem_mb)
        return True

    except Exception as e:
        print(f"Promotion failed: {e}")
        _record_decision(version, target_stage, "FAILED", approver,
                        test_auc, infer_ms, mem_mb, reason=str(e))
        return False


def promote_to_staging(version: str, approver: str = "automated_pipeline") -> bool:
    """Shortcut to promote a model to Staging."""
    return promote_model(version, "Staging", approver)


def promote_to_production(version: str, approver: str = "automated_pipeline") -> bool:
    """Shortcut to promote a Staging model to Production."""
    return promote_model(version, "Production", approver)


def get_workflow_status() -> None:
    """Show current state of all model versions in the registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.tracking.MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as e:
        print(f"Could not fetch model versions: {e}")
        return

    print(f"\n{'='*60}")
    print(f"MODEL REGISTRY STATUS — {MODEL_NAME}")
    print(f"{'='*60}")
    print(f"{'Version':<10} {'Stage':<15} {'AUC':<8} {'Created'}")
    print("-" * 60)

    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        try:
            run_info = client.get_run(v.run_id)
            auc      = run_info.data.metrics.get("test_auc", 0)
            auc_str  = f"{auc:.4f}"
        except Exception:
            auc_str = "N/A"

        created = datetime.fromtimestamp(
            v.creation_timestamp / 1000
        ).strftime("%Y-%m-%d %H:%M")

        stage_icon = {"Production": "🟢", "Staging": "🟡",
                      "Archived": "⚫", "None": "⚪"}.get(v.current_stage, "")
        print(f"  v{v.version:<8} {stage_icon} {v.current_stage:<13} {auc_str:<8} {created}")


def get_approval_history() -> None:
    """Print full approval workflow history."""
    workflow = _load_workflow()
    decisions = workflow.get("decisions", [])

    print(f"\nApproval History ({len(decisions)} decisions):")
    print(f"{'Version':<10} {'Stage':<12} {'Decision':<10} {'Approver':<25} {'Timestamp'}")
    print("-" * 75)
    for d in decisions:
        icon = "✓" if d["decision"] == "APPROVED" else "✗"
        print(f"  {icon} v{d['version']:<8} "
              f"{d['target_stage']:<12} "
              f"{d['decision']:<10} "
              f"{d['approver']:<25} "
              f"{d['timestamp'][:19]}")
        if d.get("reason"):
            print(f"    Reason: {d['reason']}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_gates(test_auc, infer_ms, mem_mb, gates, stage_name) -> bool:
    passed = True
    print(f"\nGate checks for {stage_name}:")

    auc_ok  = test_auc >= gates.get("min_auc", 0)
    inf_ok  = infer_ms <= gates.get("max_infer_ms", 9999)
    mem_ok  = mem_mb   <= gates.get("max_mem_mb", 9999)

    print(f"  AUC >= {gates.get('min_auc')}: "
          f"{'PASS' if auc_ok else 'FAIL'} ({test_auc:.4f})")
    print(f"  Infer <= {gates.get('max_infer_ms')}ms: "
          f"{'PASS' if inf_ok else 'FAIL'} ({infer_ms:.1f}ms)")
    print(f"  Memory <= {gates.get('max_mem_mb')}MB: "
          f"{'PASS' if mem_ok else 'FAIL'} ({mem_mb:.2f}MB)")

    if not (auc_ok and inf_ok and mem_ok):
        print(f"\n✗ Gate checks FAILED — promotion to {stage_name} denied")
        passed = False
    else:
        print(f"\n✓ Gate checks PASSED — promoting to {stage_name}")

    return passed


def _archive_current_production(client, new_version: str) -> None:
    """Archive existing Production model before promoting new one."""
    try:
        current = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        for v in current:
            if v.version != str(new_version):
                client.transition_model_version_stage(
                    name=MODEL_NAME, version=v.version, stage="Archived"
                )
                print(f"  Archived previous production model v{v.version}")
    except Exception as e:
        print(f"  Warning: Could not archive current production: {e}")


def _record_decision(version, target_stage, decision, approver,
                     test_auc, infer_ms, mem_mb, reason=None) -> None:
    workflow = _load_workflow()
    if "decisions" not in workflow:
        workflow["decisions"] = []

    workflow["decisions"].append({
        "version":      str(version),
        "target_stage": target_stage,
        "decision":     decision,
        "approver":     approver,
        "timestamp":    datetime.now().isoformat(),
        "metrics": {
            "test_auc":      round(test_auc, 4),
            "infer_time_ms": round(infer_ms, 1),
            "mem_peak_mb":   round(mem_mb, 2),
        },
        "reason": reason,
    })
    _save_workflow(workflow)


def _load_workflow() -> dict:
    path = Path(WORKFLOW_FILE)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_workflow(data: dict) -> None:
    path = Path(WORKFLOW_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"

    if cmd == "status":
        get_workflow_status()

    elif cmd == "history":
        get_approval_history()

    elif cmd == "promote":
        if len(sys.argv) < 3:
            print("Usage: python approval_workflow.py promote --version <n> [--stage Staging|Production]")
            sys.exit(1)
        version = sys.argv[sys.argv.index("--version") + 1] if "--version" in sys.argv else sys.argv[2]
        stage   = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "Production"
        promote_model(version, stage)

    elif cmd == "gates":
        print("\nStage Gates:")
        for stage, gates in STAGE_GATES.items():
            print(f"\n{stage}:")
            print(f"  {gates['description']}")
            print(f"  Min AUC      : {gates['min_auc']}")
            print(f"  Max Infer    : {gates['max_infer_ms']}ms")
            print(f"  Max Memory   : {gates['max_mem_mb']}MB")
    else:
        print("Usage: python approval_workflow.py [status|history|promote|gates]")
