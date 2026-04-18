"""Bottleneck prediction and recommendation engine for sprint risk analysis (M11).

Real mode: XGBoost prediction + SHAP explanation + rule-based bottleneck detection.
Mock mode: Rule-based heuristics that still produce realistic reports.

Three layers:
  1. ML prediction: risk_level + risk_score from XGBoost
  2. Rule-based detector: specific bottleneck types (always runs)
  3. Recommendation engine: actionable advice per bottleneck
"""

import logging
import random
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np

from .model import get_model, compute_sprint_features
from utils.privacy import assert_no_pii

log = logging.getLogger(__name__)

# Thresholds
_OVERLOAD_TASK_THRESHOLD = 5
_BLOCKED_RATIO_HIGH = 0.20
_SCOPE_RATIO_THRESHOLD = 1.2
_MEMBER_CAPACITY_DEFAULT = 20  # SP per member per sprint


def analyze(
    sprint_id: str,
    sprint_name: str,
    start_date: str,
    end_date: str,
    capacity: float | None,
    tasks: list[dict],
    member_workloads: dict[str, int],
    past_velocities: list[float] | None = None,
) -> dict:
    """Analyze a sprint for bottleneck risks.

    Returns a dict matching SprintAnalysisResponse schema.
    """
    assert_no_pii(
        {
            "sprint_id": sprint_id,
            "sprint_name": sprint_name,
            "tasks": tasks,
            "member_workloads": member_workloads,
        },
        context="optimizer",
    )

    om = get_model()

    # Rule-based bottleneck detection (always runs, regardless of ML)
    bottlenecks = _detect_bottlenecks(tasks, member_workloads, capacity, start_date, end_date)
    recommendations = _generate_recommendations(bottlenecks)

    if om.is_loaded and not om.is_mock:
        return _predict_with_model(
            om, sprint_id, tasks, member_workloads, capacity,
            start_date, end_date, past_velocities,
            bottlenecks, recommendations,
        )

    return _rule_based_analysis(
        om, sprint_id, sprint_name, start_date, end_date,
        capacity, tasks, member_workloads,
        bottlenecks, recommendations,
    )


def analyze_project(sprint_analyses: list[dict]) -> dict:
    """Aggregate multiple sprint analyses into a project-level summary."""
    if not sprint_analyses:
        return {
            "risk_level": "low",
            "risk_score": 0.05,
            "sprint_count": 0,
            "bottleneck_count": 0,
            "top_recommendations": [],
            "model_version": "mock-v0",
        }

    avg_score = sum(s["risk_score"] for s in sprint_analyses) / len(sprint_analyses)
    all_bottlenecks = []
    all_recs = []
    for s in sprint_analyses:
        all_bottlenecks.extend(s.get("bottlenecks", []))
        all_recs.extend(s.get("recommendations", []))

    # Deduplicate recommendations
    seen = set()
    unique_recs = []
    for r in all_recs:
        if r["action"] not in seen:
            seen.add(r["action"])
            unique_recs.append(r)

    return {
        "risk_level": _score_to_level(avg_score),
        "risk_score": round(avg_score, 2),
        "sprint_count": len(sprint_analyses),
        "bottleneck_count": len(all_bottlenecks),
        "top_recommendations": unique_recs[:5],
        "model_version": sprint_analyses[0].get("model_version", "mock-v0"),
    }


# ---------------------------------------------------------------------------
# Real model prediction
# ---------------------------------------------------------------------------

def _predict_with_model(
    om, sprint_id, tasks, member_workloads, capacity,
    start_date, end_date, past_velocities,
    bottlenecks, recommendations,
) -> dict:
    """Run XGBoost + optional SHAP for risk prediction."""
    # 1. Compute feature vector
    features = compute_sprint_features(
        tasks, member_workloads, capacity, start_date, end_date, past_velocities
    )

    # 2. Scale
    features_scaled = om.scaler.transform(features.reshape(1, -1))

    # 3. ML prediction
    probas = om.model.predict_proba(features_scaled)[0]
    predicted_class = int(np.argmax(probas))
    risk_score = float(np.max(probas))

    # Decode risk level
    if om.label_encoder is not None:
        risk_level = om.label_encoder.inverse_transform([predicted_class])[0]
    else:
        risk_levels = ["critical", "high", "low", "medium"]  # alphabetical default
        risk_level = risk_levels[predicted_class] if predicted_class < len(risk_levels) else "medium"

    # 4. SHAP explanation (top 3 contributing features)
    shap_explanation = None
    if om.explainer is not None:
        try:
            shap_values = om.explainer.shap_values(features_scaled)
            # For multiclass, get SHAP for the predicted class
            if isinstance(shap_values, list):
                class_shap = shap_values[predicted_class][0]
            else:
                class_shap = shap_values[0]

            # Top 3 features by absolute SHAP value
            feature_names = om.feature_names
            abs_shap = np.abs(class_shap)
            top_indices = np.argsort(abs_shap)[::-1][:3]
            shap_explanation = [
                {
                    "feature": feature_names[idx],
                    "value": float(features[idx]),
                    "impact": round(float(class_shap[idx]), 4),
                }
                for idx in top_indices
            ]
        except Exception as exc:
            log.warning("SHAP explanation failed: %s", exc)

    # Build response
    result = {
        "sprint_id": sprint_id,
        "risk_level": risk_level,
        "risk_score": round(risk_score, 2),
        "bottlenecks": bottlenecks,
        "recommendations": recommendations,
        "model_version": om.version,
    }

    if shap_explanation:
        # Attach SHAP info to bottlenecks as context
        for bn in result["bottlenecks"]:
            bn["_shap_context"] = shap_explanation

    return result


# ---------------------------------------------------------------------------
# Rule-based / mock analysis
# ---------------------------------------------------------------------------

def _rule_based_analysis(
    om, sprint_id, sprint_name, start_date, end_date,
    capacity, tasks, member_workloads,
    bottlenecks, recommendations,
) -> dict:
    """Fall back to rule-based risk scoring when ML model is unavailable."""
    log.info("Rule-based analysis for sprint %s (%s)", sprint_id, sprint_name)

    # Compute risk score from bottleneck severities
    severity_weights = {"low": 0.1, "medium": 0.25, "high": 0.5, "critical": 0.8}
    if bottlenecks:
        total_severity = sum(severity_weights.get(b["severity"], 0.1) for b in bottlenecks)
        risk_score = min(1.0, total_severity / max(len(bottlenecks), 1))
    else:
        risk_score = round(random.uniform(0.05, 0.15), 2)

    # Timeline pressure boost
    try:
        now = datetime.now(timezone.utc)
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        total_days = (end - start).days or 1
        remaining = max(0, (end - now).days)
        elapsed_ratio = 1.0 - (remaining / total_days)

        incomplete = sum(1 for t in tasks if t.get("status") not in ("done",))
        total = len(tasks) or 1
        if elapsed_ratio > 0.7 and (incomplete / total) > 0.5:
            risk_score = min(1.0, risk_score + 0.15)
    except (ValueError, TypeError):
        pass

    risk_score = round(risk_score, 2)

    return {
        "sprint_id": sprint_id,
        "risk_level": _score_to_level(risk_score),
        "risk_score": risk_score,
        "bottlenecks": bottlenecks,
        "recommendations": recommendations,
        "model_version": om.version,
    }


# ---------------------------------------------------------------------------
# Bottleneck detection (rule-based — always runs)
# ---------------------------------------------------------------------------

def _detect_bottlenecks(
    tasks: list[dict],
    member_workloads: dict[str, int],
    capacity: float | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """Identify specific bottleneck types in a sprint.

    These deterministic rules always fire regardless of ML prediction.
    """
    bottlenecks: list[dict] = []
    if not tasks:
        return bottlenecks

    total_tasks = len(tasks)

    # 1. Member overload
    member_tasks: dict[str, list[str]] = defaultdict(list)
    for t in tasks:
        aid = t.get("assignee_id")
        if aid and t.get("status") not in ("done",):
            member_tasks[aid].append(t.get("task_id", t.get("id", "")))

    overloaded_count = 0
    for uid, task_ids in member_tasks.items():
        if len(task_ids) > _OVERLOAD_TASK_THRESHOLD:
            overloaded_count += 1
            severity = "critical" if len(task_ids) > _OVERLOAD_TASK_THRESHOLD + 3 else "high" if len(task_ids) > _OVERLOAD_TASK_THRESHOLD + 1 else "medium"
            bottlenecks.append({
                "type": "member_overload",
                "description": f"Member {uid[:8]} has {len(task_ids)} active tasks (threshold: {_OVERLOAD_TASK_THRESHOLD})",
                "affected_task_ids": task_ids,
                "severity": severity,
            })

    # 2. Blocked tasks
    blocked = [t for t in tasks if t.get("status") == "blocked"]
    if len(blocked) > 2:
        blocked_ratio = len(blocked) / total_tasks
        severity = "critical" if len(blocked) > 5 else "high" if len(blocked) > 3 else "medium"
        bottlenecks.append({
            "type": "blocked_tasks",
            "description": f"{len(blocked)} of {total_tasks} tasks are blocked ({blocked_ratio:.0%})",
            "affected_task_ids": [t.get("task_id", t.get("id", "")) for t in blocked],
            "severity": severity,
        })

    # 3. Unassigned critical tasks
    unassigned_critical = [
        t for t in tasks
        if not t.get("assignee_id")
        and t.get("status") != "done"
        and t.get("priority") in ("critical", "urgent", "high")
    ]
    if unassigned_critical:
        bottlenecks.append({
            "type": "unassigned_critical",
            "description": f"{len(unassigned_critical)} high-priority task{'s' if len(unassigned_critical) > 1 else ''} unassigned",
            "affected_task_ids": [t.get("task_id", t.get("id", "")) for t in unassigned_critical],
            "severity": "high",
        })

    # 4. Velocity lag (deadline risk check)
    try:
        if end_date and start_date:
            now = datetime.now(timezone.utc)
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            sprint_length = max(1, (end_dt - start_dt).days)
            days_remaining = max(0, (end_dt - now).days)
            elapsed_frac = (sprint_length - days_remaining) / sprint_length

            done_count = sum(1 for t in tasks if t.get("status") == "done")
            completion_rate = done_count / total_tasks
            expected = max(elapsed_frac, 0.01)
            velocity_vs_plan = completion_rate / expected

            if velocity_vs_plan < 0.6:
                pct_behind = int((1 - velocity_vs_plan) * 100)
                bottlenecks.append({
                    "type": "velocity_lag",
                    "description": f"Sprint velocity is {pct_behind}% behind planned pace",
                    "affected_task_ids": [],
                    "severity": "high" if velocity_vs_plan < 0.4 else "medium",
                })

            # 5. Deadline risk
            if days_remaining < 2 and completion_rate < 0.7:
                remaining_pct = int((1 - completion_rate) * 100)
                bottlenecks.append({
                    "type": "deadline_risk",
                    "description": f"Sprint ends in {days_remaining} day{'s' if days_remaining != 1 else ''} with {remaining_pct}% work remaining",
                    "affected_task_ids": [],
                    "severity": "critical",
                })
    except (ValueError, TypeError):
        pass

    # 6. Scope creep
    if capacity and capacity > 0:
        total_sp = sum(t.get("story_points") or 0 for t in tasks)
        if total_sp > capacity * _SCOPE_RATIO_THRESHOLD:
            overcommit = int(((total_sp / capacity) - 1) * 100)
            bottlenecks.append({
                "type": "scope_creep",
                "description": f"Sprint is {overcommit}% over capacity ({total_sp} SP vs {capacity} capacity)",
                "affected_task_ids": [],
                "severity": "high" if total_sp > capacity * 1.4 else "medium",
            })

    # 7. Workload imbalance
    workloads = list(member_workloads.values()) if member_workloads else []
    if workloads and len(workloads) > 1:
        mean_wl = np.mean(workloads)
        if mean_wl > 0:
            std_ratio = float(np.std(workloads) / mean_wl)
            if std_ratio > 0.5:
                bottlenecks.append({
                    "type": "workload_imbalance",
                    "description": "Significant workload imbalance across team members",
                    "affected_task_ids": [],
                    "severity": "medium",
                })

    # 8. Overdue tasks
    now = datetime.now(timezone.utc)
    overdue = []
    for t in tasks:
        if t.get("status") == "done":
            continue
        due = t.get("due_date")
        if due:
            try:
                due_dt = datetime.fromisoformat(due.replace("Z", "+00:00"))
                if due_dt < now:
                    overdue.append(t)
            except (ValueError, TypeError):
                pass
    if len(overdue) > 3:
        bottlenecks.append({
            "type": "overdue_tasks",
            "description": f"{len(overdue)} tasks are past their due date",
            "affected_task_ids": [t.get("task_id", t.get("id", "")) for t in overdue],
            "severity": "high" if len(overdue) > 5 else "medium",
        })

    # 9. Unassigned tasks (general)
    unassigned_all = [t for t in tasks if not t.get("assignee_id") and t.get("status") != "done"]
    if len(unassigned_all) > 3:
        bottlenecks.append({
            "type": "unassigned_tasks",
            "description": f"{len(unassigned_all)} tasks have no assignee",
            "affected_task_ids": [t.get("task_id", t.get("id", "")) for t in unassigned_all],
            "severity": "medium" if len(unassigned_all) <= 5 else "high",
        })

    return bottlenecks


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

def _generate_recommendations(bottlenecks: list[dict]) -> list[dict]:
    """Map bottleneck types to actionable recommendations."""
    recommendations: list[dict] = []

    for bn in bottlenecks:
        bn_type = bn["type"]
        severity = bn["severity"]

        if bn_type == "member_overload":
            recommendations.append({
                "action": "Redistribute tasks from overloaded members to team members with available capacity",
                "reason": bn["description"],
                "priority": "high",
            })

        elif bn_type == "blocked_tasks":
            recommendations.append({
                "action": "Schedule a blocker resolution meeting to unblock stuck tasks",
                "reason": bn["description"],
                "priority": "high" if severity in ("high", "critical") else "medium",
            })

        elif bn_type == "unassigned_critical":
            recommendations.append({
                "action": "Immediately assign all unassigned high-priority tasks",
                "reason": bn["description"],
                "priority": "high",
            })

        elif bn_type == "velocity_lag":
            recommendations.append({
                "action": "Consider reducing sprint scope — move low-priority tasks to backlog",
                "reason": bn["description"],
                "priority": "high" if severity == "high" else "medium",
            })

        elif bn_type == "deadline_risk":
            recommendations.append({
                "action": "Sprint at risk — escalate to project lead and prioritize must-have items",
                "reason": bn["description"],
                "priority": "high",
            })

        elif bn_type == "scope_creep":
            recommendations.append({
                "action": "Move lowest-priority tasks to backlog to bring sprint within capacity",
                "reason": bn["description"],
                "priority": "medium",
            })

        elif bn_type == "workload_imbalance":
            recommendations.append({
                "action": "Rebalance task assignments to distribute workload more evenly",
                "reason": bn["description"],
                "priority": "medium",
            })

        elif bn_type == "overdue_tasks":
            recommendations.append({
                "action": "Review overdue tasks — update due dates or reassign to unblock progress",
                "reason": bn["description"],
                "priority": "high" if severity == "high" else "medium",
            })

        elif bn_type == "unassigned_tasks":
            recommendations.append({
                "action": "Use AI smart assignment to assign unowned tasks to available team members",
                "reason": bn["description"],
                "priority": "medium",
            })

    # Deduplicate
    seen = set()
    unique = []
    for r in recommendations:
        if r["action"] not in seen:
            seen.add(r["action"])
            unique.append(r)

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    unique.sort(key=lambda r: priority_order.get(r["priority"], 2))

    return unique


def _score_to_level(score: float) -> str:
    """Convert numeric risk score to risk level string."""
    if score >= 0.75:
        return "critical"
    if score >= 0.50:
        return "high"
    if score >= 0.25:
        return "medium"
    return "low"
