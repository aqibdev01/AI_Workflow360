"""XGBoost bottleneck predictor for sprint risk analysis (M11).

Loads:
  - optimizer_model.pkl     (XGBClassifier)
  - optimizer_scaler.pkl    (StandardScaler)
  - optimizer_explainer.pkl (SHAP TreeExplainer — optional, for explanations)
  - optimizer_label_encoder.pkl (LabelEncoder for risk level classes)

Falls back to mock mode (rule-based heuristics) if pkl files are missing.
"""

import os
import json
import logging
from datetime import datetime, timezone

import numpy as np

log = logging.getLogger(__name__)

_model = None
_scaler = None
_explainer = None
_label_encoder = None
_feature_names: list[str] = []
_mock_mode: bool = True
_model_version: str = "mock-v0"

# Default feature names matching training script
_DEFAULT_FEATURE_NAMES = [
    "days_remaining",
    "completion_rate",
    "story_points_completed",
    "story_points_remaining",
    "capacity_utilization",
    "past_velocity_avg",
    "blocked_task_count",
    "critical_task_count",
    "unassigned_task_count",
    "overdue_task_count",
    "tasks_without_estimates",
    "max_member_workload",
    "workload_std_dev",
    "avg_member_utilization",
    "overloaded_member_count",
    "velocity_trend",
    "velocity_vs_plan",
    "scope_change_ratio",
]


class OptimizerModel:
    """Wrapper around the trained XGBoost sprint risk model."""

    def __init__(self) -> None:
        self.model = _model
        self.scaler = _scaler
        self.explainer = _explainer
        self.label_encoder = _label_encoder
        self.feature_names = _feature_names or _DEFAULT_FEATURE_NAMES
        self.version = _model_version

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None

    @property
    def is_mock(self) -> bool:
        return _mock_mode


def compute_sprint_features(
    tasks: list[dict],
    member_workloads: dict[str, int],
    capacity: float | None,
    start_date: str,
    end_date: str,
    past_velocities: list[float] | None = None,
) -> np.ndarray:
    """Compute the 18-dim feature vector from sprint data.

    Returns a numpy array matching the training feature order.
    """
    now = datetime.now(timezone.utc)
    total_tasks = len(tasks) or 1

    # Parse dates
    try:
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        days_remaining = max(0.0, (end_dt - now).total_seconds() / 86400)
        sprint_length = max(1.0, (end_dt - start_dt).total_seconds() / 86400)
    except (ValueError, TypeError):
        days_remaining = 7.0
        sprint_length = 14.0

    # Task status counts
    done_tasks = [t for t in tasks if t.get("status") == "done"]
    blocked_tasks = [t for t in tasks if t.get("status") == "blocked"]
    critical_tasks = [t for t in tasks if t.get("priority") in ("critical", "urgent", "high")]
    unassigned_tasks = [t for t in tasks if not t.get("assignee_id") and t.get("status") != "done"]
    no_estimate_tasks = [t for t in tasks if not t.get("story_points")]

    # Overdue tasks
    overdue_tasks = []
    for t in tasks:
        if t.get("status") == "done":
            continue
        due = t.get("due_date")
        if due:
            try:
                due_dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
                if due_dt < now:
                    overdue_tasks.append(t)
            except (ValueError, TypeError):
                pass

    # Story points
    sp_completed = sum(t.get("story_points") or 0 for t in done_tasks)
    sp_remaining = sum(t.get("story_points") or 0 for t in tasks if t.get("status") != "done")
    total_sp = sp_completed + sp_remaining

    completion_rate = len(done_tasks) / total_tasks

    # Capacity
    cap = capacity or 40.0
    capacity_utilization = total_sp / cap if cap > 0 else 0.0

    # Past velocity
    vels = past_velocities or []
    past_velocity_avg = float(np.mean(vels)) if vels else 30.0

    # Velocity trend
    if len(vels) >= 2:
        velocity_trend = (vels[0] - vels[-1]) / max(vels[-1], 1.0)
    else:
        velocity_trend = 0.0

    # Velocity vs plan
    elapsed_fraction = (sprint_length - days_remaining) / sprint_length if sprint_length > 0 else 0.9
    expected_completion = max(elapsed_fraction, 0.01)
    velocity_vs_plan = min(2.0, completion_rate / expected_completion)

    # Member workload stats
    workloads = list(member_workloads.values()) if member_workloads else [0]
    max_member_workload = float(max(workloads)) if workloads else 0.0
    workload_std = float(np.std(workloads) / max(np.mean(workloads), 1.0)) if workloads else 0.0
    member_capacity = cap / max(len(workloads), 1)
    avg_util = float(np.mean(workloads)) / max(member_capacity, 1.0) if workloads else 0.0
    overloaded_count = sum(1 for w in workloads if w > member_capacity * 0.8)

    # Scope change (not available at inference time — use 0)
    scope_change = 0.0

    features = np.array([
        days_remaining,
        completion_rate,
        sp_completed,
        sp_remaining,
        capacity_utilization,
        past_velocity_avg,
        len(blocked_tasks),
        len(critical_tasks),
        len(unassigned_tasks),
        len(overdue_tasks),
        len(no_estimate_tasks),
        max_member_workload,
        workload_std,
        avg_util,
        overloaded_count,
        velocity_trend,
        velocity_vs_plan,
        scope_change,
    ], dtype=np.float32)

    return features


def load_model() -> OptimizerModel:
    """Attempt to load XGBoost model, scaler, and optionally SHAP explainer.

    Falls back to mock mode if required pkl files are missing.
    """
    global _model, _scaler, _explainer, _label_encoder, _feature_names
    global _mock_mode, _model_version

    if _model is not None:
        return OptimizerModel()

    model_dir = os.getenv("MODEL_DIR", "./model_weights")
    model_path = os.path.join(model_dir, "optimizer")

    model_file = os.path.join(model_path, "optimizer_model.pkl")
    scaler_file = os.path.join(model_path, "optimizer_scaler.pkl")

    if not os.path.isfile(model_file) or not os.path.isfile(scaler_file):
        missing = []
        if not os.path.isfile(model_file):
            missing.append("optimizer_model.pkl")
        if not os.path.isfile(scaler_file):
            missing.append("optimizer_scaler.pkl")
        log.warning(
            "Optimizer model files missing at %s (%s) — running in MOCK mode",
            model_path,
            ", ".join(missing),
        )
        _mock_mode = True
        _model_version = "mock-v0"
        return OptimizerModel()

    try:
        import joblib

        # Load XGBoost model
        log.info("Loading optimizer model from %s...", model_file)
        _model = joblib.load(model_file)
        log.info("XGBoost model loaded: %s", type(_model).__name__)

        # Load scaler
        log.info("Loading scaler from %s...", scaler_file)
        _scaler = joblib.load(scaler_file)

        # Load label encoder (optional)
        le_file = os.path.join(model_path, "optimizer_label_encoder.pkl")
        if os.path.isfile(le_file):
            _label_encoder = joblib.load(le_file)
            log.info("Label encoder loaded: classes=%s", list(_label_encoder.classes_))

        # Load SHAP explainer (optional)
        explainer_file = os.path.join(model_path, "optimizer_explainer.pkl")
        if os.path.isfile(explainer_file):
            _explainer = joblib.load(explainer_file)
            log.info("SHAP explainer loaded")
        else:
            log.info("SHAP explainer not found — explanations will be unavailable")

        # Load metadata for feature names and version
        metadata_file = os.path.join(model_path, "training_metadata.json")
        if os.path.isfile(metadata_file):
            with open(metadata_file) as f:
                meta = json.load(f)
            _feature_names = meta.get("feature_names", _DEFAULT_FEATURE_NAMES)
            _model_version = meta.get("model_version", "optimizer-v1")
        else:
            _feature_names = _DEFAULT_FEATURE_NAMES
            _model_version = "optimizer-v1"

        _mock_mode = False
        log.info("Optimizer model ready: %s", _model_version)

    except ImportError as exc:
        log.warning("Missing dependency for optimizer (%s) — MOCK mode", exc)
        _model = None
        _scaler = None
        _mock_mode = True
        _model_version = "mock-v0"

    except Exception as exc:
        log.warning("Failed to load optimizer model (%s) — MOCK mode", exc)
        _model = None
        _scaler = None
        _mock_mode = True
        _model_version = "mock-v0"

    return OptimizerModel()


def get_model() -> OptimizerModel:
    """Return the singleton (may be in mock mode)."""
    return OptimizerModel()
