#!/usr/bin/env python3
"""
=============================================================================
Sprint Bottleneck Predictor Training — XGBoost + SHAP (M11)
=============================================================================
Workflow360 FYP — AI Module Training

This script trains an XGBoost multiclass classifier to predict sprint
bottleneck risk (low/medium/high/critical) from sprint and task features.
Includes a rule-based bottleneck detector and recommendation engine
layered on top of the ML model.

Trainable on CPU in minutes. No GPU required.

Architecture:
  1. XGBoost classifier predicts overall risk level (4 classes)
  2. Rule-based bottleneck detector identifies specific issues
  3. SHAP explainer shows WHY a sprint is at risk
  4. Recommendation engine maps bottleneck types to actionable advice

Usage:
  python train_optimizer.py          # local run
  # Or run cell-by-cell in Google Colab
=============================================================================
"""

# ============================================================================
# MARKDOWN: # Sprint Bottleneck Predictor — XGBoost + SHAP (M11)
#
# Trains an **XGBoost multiclass classifier** to predict sprint bottleneck
# risk levels, with a **rule-based bottleneck detector** and
# **recommendation engine** layered on top.
#
# - **Input:** 18-feature sprint snapshot (task distribution, workload, velocity)
# - **Output:** Risk level (low/medium/high/critical) + specific bottlenecks + recommendations
# - **Hardware:** CPU only (~5-10 minutes training)
# - **Target:** >82% accuracy, <10ms inference per sprint
# ============================================================================


# ============================================================================
# SECTION 1: Setup & Dependencies
# ============================================================================
# MARKDOWN: ## 1. Setup & Dependencies
# Install XGBoost, scikit-learn, SHAP for explainability, and matplotlib
# for feature importance visualization.

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install("xgboost>=2.0.3")
install("scikit-learn>=1.5.0")
install("pandas>=2.2.0")
install("numpy>=1.26.0")
install("joblib>=1.4.0")
install("matplotlib>=3.8.0")
install("shap>=0.45.0")

import os
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Dependencies loaded successfully")
print(f"XGBoost version: {xgb.__version__}")
print(f"SHAP version: {shap.__version__}")


# ============================================================================
# SECTION 2: Synthetic Dataset Generation
# ============================================================================
# MARKDOWN: ## 2. Synthetic Dataset Generation
#
# We generate 8000 sprint snapshots, each with 18 features capturing sprint
# health across 4 dimensions:
#
# | Dimension | Features |
# |-----------|----------|
# | Sprint-level | days_remaining, completion_rate, SP completed/remaining, capacity util, past velocity |
# | Task distribution | blocked, critical, unassigned, overdue, no-estimate counts |
# | Member workload | max workload, std dev, avg utilization, overloaded count |
# | Velocity trend | trend direction, pace vs plan |
#
# **Labeling rules:**
# - **critical:** completion <20% with <3 days left, OR >3 overloaded members
# - **high:** completion <40% with <5 days left, OR >4 blocked tasks
# - **medium:** velocity vs plan <0.7, OR workload std dev >0.4
# - **low:** everything else

print("\n" + "=" * 60)
print("SECTION 2: Generating synthetic sprint data")
print("=" * 60)

FEATURE_NAMES = [
    # Sprint-level
    "days_remaining",
    "completion_rate",
    "story_points_completed",
    "story_points_remaining",
    "capacity_utilization",
    "past_velocity_avg",
    # Task distribution
    "blocked_task_count",
    "critical_task_count",
    "unassigned_task_count",
    "overdue_task_count",
    "tasks_without_estimates",
    # Member workload
    "max_member_workload",
    "workload_std_dev",
    "avg_member_utilization",
    "overloaded_member_count",
    # Velocity trend
    "velocity_trend",
    "velocity_vs_plan",
    # Derived
    "scope_change_ratio",
]

RISK_LEVELS = ["low", "medium", "high", "critical"]


def label_sprint(f: dict) -> str:
    """Apply deterministic labeling rules to assign a risk level."""
    # Critical conditions
    if f["completion_rate"] < 0.2 and f["days_remaining"] < 3:
        return "critical"
    if f["overloaded_member_count"] > 3:
        return "critical"
    if f["blocked_task_count"] > 6 and f["days_remaining"] < 4:
        return "critical"

    # High conditions
    if f["completion_rate"] < 0.4 and f["days_remaining"] < 5:
        return "high"
    if f["blocked_task_count"] > 4:
        return "high"
    if f["overloaded_member_count"] > 2 and f["velocity_vs_plan"] < 0.6:
        return "high"
    if f["critical_task_count"] > 3 and f["unassigned_task_count"] > 2:
        return "high"

    # Medium conditions
    if f["velocity_vs_plan"] < 0.7:
        return "medium"
    if f["workload_std_dev"] > 0.4:
        return "medium"
    if f["overdue_task_count"] > 3:
        return "medium"
    if f["capacity_utilization"] > 1.1:
        return "medium"
    if f["unassigned_task_count"] > 3:
        return "medium"

    return "low"


def generate_sprint(bias: str | None = None) -> dict:
    """Generate a single sprint snapshot with realistic feature values.

    Args:
        bias: optionally bias toward a specific risk level to balance the dataset.
    """
    # Sprint-level features
    sprint_length = random.choice([7, 10, 14])  # days
    total_tasks = random.randint(8, 40)
    total_sp = random.randint(20, 80)
    capacity = random.randint(30, 80)
    team_size = random.randint(3, 10)

    if bias == "critical":
        days_remaining = random.uniform(0.5, 3.0)
        completion_rate = random.uniform(0.05, 0.25)
        overloaded_count = random.randint(2, 5)
        blocked = random.randint(3, 8)
    elif bias == "high":
        days_remaining = random.uniform(1.5, 5.5)
        completion_rate = random.uniform(0.15, 0.45)
        overloaded_count = random.randint(1, 3)
        blocked = random.randint(2, 7)
    elif bias == "medium":
        days_remaining = random.uniform(3.0, 8.0)
        completion_rate = random.uniform(0.3, 0.65)
        overloaded_count = random.randint(0, 2)
        blocked = random.randint(0, 4)
    else:  # low
        days_remaining = random.uniform(5.0, 12.0)
        completion_rate = random.uniform(0.5, 0.95)
        overloaded_count = random.randint(0, 1)
        blocked = random.randint(0, 2)

    sp_completed = round(total_sp * completion_rate, 1)
    sp_remaining = round(total_sp - sp_completed, 1)
    capacity_util = round(total_sp / max(capacity, 1), 2)

    # Past velocity
    past_velocities = [random.uniform(15, 60) for _ in range(3)]
    past_velocity_avg = round(np.mean(past_velocities), 1)

    # Task distribution
    critical_tasks = random.randint(0, max(1, total_tasks // 4))
    unassigned = random.randint(0, max(1, total_tasks // 3))
    overdue = random.randint(0, max(1, int(total_tasks * (1 - completion_rate) * 0.5)))
    no_estimates = random.randint(0, max(1, total_tasks // 5))

    # Member workload
    member_workloads = [random.uniform(5, 30) for _ in range(team_size)]
    max_workload = round(max(member_workloads), 1)
    workload_std = round(float(np.std(member_workloads) / max(np.mean(member_workloads), 1)), 3)
    avg_util = round(float(np.mean(member_workloads)) / max(capacity / team_size, 1), 3)

    # Velocity trend
    if len(past_velocities) >= 2:
        trend = round((past_velocities[-1] - past_velocities[0]) / max(past_velocities[0], 1), 3)
    else:
        trend = 0.0

    # Velocity vs plan: are we on pace?
    if days_remaining > 0 and sprint_length > 0:
        elapsed_fraction = (sprint_length - days_remaining) / sprint_length
        expected_completion = elapsed_fraction
        velocity_vs_plan = round(completion_rate / max(expected_completion, 0.01), 3)
        velocity_vs_plan = min(2.0, velocity_vs_plan)
    else:
        velocity_vs_plan = round(completion_rate / 0.9, 3)

    # Scope change ratio
    scope_change = round(random.uniform(-0.1, 0.3), 3)

    features = {
        "days_remaining": round(days_remaining, 1),
        "completion_rate": round(completion_rate, 3),
        "story_points_completed": sp_completed,
        "story_points_remaining": sp_remaining,
        "capacity_utilization": capacity_util,
        "past_velocity_avg": past_velocity_avg,
        "blocked_task_count": blocked,
        "critical_task_count": critical_tasks,
        "unassigned_task_count": unassigned,
        "overdue_task_count": overdue,
        "tasks_without_estimates": no_estimates,
        "max_member_workload": max_workload,
        "workload_std_dev": workload_std,
        "avg_member_utilization": avg_util,
        "overloaded_member_count": overloaded_count,
        "velocity_trend": trend,
        "velocity_vs_plan": velocity_vs_plan,
        "scope_change_ratio": scope_change,
    }

    return features


def generate_dataset(n: int = 8000) -> pd.DataFrame:
    """Generate n sprint examples with balanced class distribution."""
    # Generate with class bias for balance
    per_class = n // 4
    examples = []

    for risk_class in RISK_LEVELS:
        count = 0
        attempts = 0
        while count < per_class and attempts < per_class * 10:
            features = generate_sprint(bias=risk_class)
            label = label_sprint(features)
            # Accept if label matches bias (or close enough after many tries)
            if label == risk_class or attempts > per_class * 5:
                features["risk_level"] = label
                examples.append(features)
                count += 1
            attempts += 1

    random.shuffle(examples)
    df = pd.DataFrame(examples)

    # Add some noise: flip 5% of labels to make it harder
    flip_indices = random.sample(range(len(df)), int(0.05 * len(df)))
    for idx in flip_indices:
        current = df.at[idx, "risk_level"]
        alternatives = [r for r in RISK_LEVELS if r != current]
        # Prefer adjacent risk levels
        if current == "low":
            df.at[idx, "risk_level"] = "medium"
        elif current == "critical":
            df.at[idx, "risk_level"] = "high"
        else:
            df.at[idx, "risk_level"] = random.choice(alternatives)

    return df


print("Generating 8000 sprint examples...")
df = generate_dataset(8000)

print(f"\nDataset shape: {df.shape}")
print(f"\nRisk level distribution:")
print(df["risk_level"].value_counts().sort_index().to_string())
print(f"\nFeature statistics:")
print(df[FEATURE_NAMES].describe().round(3).to_string())


# ============================================================================
# SECTION 3: Model Training
# ============================================================================
# MARKDOWN: ## 3. Model Training
#
# We train an **XGBoost multiclass classifier** (4 classes) with
# hyperparameter tuning via GridSearchCV:
# - n_estimators: [100, 200, 300]
# - max_depth: [3, 4, 5]
# - learning_rate: [0.05, 0.1, 0.2]
#
# Evaluation: 5-fold stratified CV, accuracy, macro F1, confusion matrix.
# Target: >82% accuracy.

print("\n" + "=" * 60)
print("SECTION 3: Model training")
print("=" * 60)

# Prepare features and labels
X = df[FEATURE_NAMES].values.astype(np.float32)
le = LabelEncoder()
y = le.fit_transform(df["risk_level"])
class_names = le.classes_.tolist()

print(f"Classes: {class_names}")
print(f"Label encoding: {dict(zip(class_names, le.transform(class_names)))}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y,
)
print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# Hyperparameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1, 0.2],
}

base_model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(class_names),
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=SEED,
    verbosity=0,
)

print(f"\nRunning GridSearchCV with {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])} parameter combinations...")
print("This may take 3-8 minutes on CPU...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

t0 = time.time()
grid_search = GridSearchCV(
    base_model,
    param_grid,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=0,
    refit=True,
)
grid_search.fit(X_train, y_train)
search_time = time.time() - t0

print(f"\nGrid search completed in {search_time:.1f}s")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1 (macro): {grid_search.best_score_:.4f}")

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average="macro")
print(f"\n--- Test Set Results ---")
print(f"Accuracy: {test_acc:.4f}")
print(f"Macro F1: {test_f1:.4f}")

if test_acc >= 0.82:
    print(f"TARGET MET: accuracy {test_acc:.4f} >= 0.82")
else:
    print(f"TARGET NOT MET: accuracy {test_acc:.4f} < 0.82")

print(f"\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=class_names))

print(f"\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
print(cm_df.to_string())

# XGBoost built-in feature importance
print(f"\n--- XGBoost Feature Importance (gain) ---")
importance = best_model.feature_importances_
for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])[:10]:
    bar = "#" * int(imp * 80)
    print(f"  {name:30s} {imp:.4f} {bar}")


# ============================================================================
# SECTION 4: Explainability (SHAP)
# ============================================================================
# MARKDOWN: ## 4. Explainability with SHAP
#
# SHAP (SHapley Additive exPlanations) shows which features drive each
# prediction. We compute a SHAP explainer that can be used at inference
# time to explain WHY a specific sprint is flagged as high/critical risk.

print("\n" + "=" * 60)
print("SECTION 4: SHAP explainability")
print("=" * 60)

print("Computing SHAP values (this may take 1-2 minutes)...")
t0 = time.time()

# Use a sample for SHAP computation (faster than full dataset)
X_shap_sample = X_test[:500]
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap_sample)

shap_time = time.time() - t0
print(f"SHAP computation completed in {shap_time:.1f}s")

# Feature importance from SHAP (mean |SHAP| across all classes)
if isinstance(shap_values, list):
    # Multi-class: average across classes
    mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
else:
    mean_shap = np.abs(shap_values).mean(axis=0)

print(f"\n--- SHAP Feature Importance (top 10) ---")
shap_importance = sorted(zip(FEATURE_NAMES, mean_shap), key=lambda x: -x[1])
for name, imp in shap_importance[:10]:
    bar = "#" * int(imp * 100)
    print(f"  {name:30s} {imp:.4f} {bar}")

# Plot SHAP summary
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    # Use class 0 (or combined) for summary plot
    if isinstance(shap_values, list):
        shap.summary_plot(
            shap_values[0],
            X_shap_sample,
            feature_names=FEATURE_NAMES,
            show=False,
            plot_size=(10, 6),
        )
    else:
        shap.summary_plot(
            shap_values,
            X_shap_sample,
            feature_names=FEATURE_NAMES,
            show=False,
            plot_size=(10, 6),
        )
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png", dpi=150, bbox_inches="tight")
    print("\nSaved SHAP plot: shap_feature_importance.png")
    plt.close()
except Exception as e:
    print(f"Could not create SHAP plot: {e}")

# Bar chart of top 10 features
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    top_features = shap_importance[:10]
    names = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    bars = ax.barh(range(len(names)), values, color="#7c3aed")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 10 Features — Sprint Risk Prediction")
    plt.tight_layout()
    plt.savefig("top_features_bar.png", dpi=150, bbox_inches="tight")
    print("Saved feature importance bar chart: top_features_bar.png")
    plt.close()
except Exception as e:
    print(f"Could not create bar chart: {e}")


# ============================================================================
# SECTION 5: Bottleneck Detector (Rule-Based Layer)
# ============================================================================
# MARKDOWN: ## 5. Bottleneck Detector — Rule-Based Layer
#
# Deterministic rules that always fire regardless of the ML risk score.
# These identify **specific** bottleneck types that the ML model predicts
# the risk of. The rules catch known patterns that should always be flagged.

print("\n" + "=" * 60)
print("SECTION 5: Bottleneck detector rules")
print("=" * 60)


def detect_bottlenecks(sprint_data: dict) -> list[dict]:
    """Identify specific bottleneck types in a sprint snapshot.

    Always runs regardless of ML risk prediction. Returns a list of
    bottleneck dicts with type, description, severity, and affected context.

    Args:
        sprint_data: dict with sprint feature values + optional task/member details.
    """
    bottlenecks = []

    # 1. Member overload
    overloaded = sprint_data.get("overloaded_member_count", 0)
    if overloaded > 0:
        severity = "critical" if overloaded > 3 else "high" if overloaded > 1 else "medium"
        bottlenecks.append({
            "type": "member_overload",
            "description": f"{overloaded} team member{'s' if overloaded > 1 else ''} "
                           f"{'are' if overloaded > 1 else 'is'} over 80% capacity",
            "severity": severity,
            "affected_task_ids": sprint_data.get("overloaded_member_task_ids", []),
        })

    # 2. Blocked tasks
    blocked = sprint_data.get("blocked_task_count", 0)
    if blocked > 2:
        severity = "critical" if blocked > 5 else "high" if blocked > 3 else "medium"
        bottlenecks.append({
            "type": "blocked_tasks",
            "description": f"{blocked} tasks are blocked and need resolution",
            "severity": severity,
            "affected_task_ids": sprint_data.get("blocked_task_ids", []),
        })

    # 3. Unassigned critical tasks
    unassigned_critical = sprint_data.get("unassigned_critical_tasks", 0)
    if unassigned_critical is None:
        # Estimate from features
        critical = sprint_data.get("critical_task_count", 0)
        unassigned = sprint_data.get("unassigned_task_count", 0)
        unassigned_critical = min(critical, unassigned)
    if unassigned_critical > 0:
        bottlenecks.append({
            "type": "unassigned_critical",
            "description": f"{unassigned_critical} critical task{'s' if unassigned_critical > 1 else ''} "
                           f"{'are' if unassigned_critical > 1 else 'is'} unassigned",
            "severity": "high",
            "affected_task_ids": sprint_data.get("unassigned_critical_task_ids", []),
        })

    # 4. Velocity lag
    velocity_vs_plan = sprint_data.get("velocity_vs_plan", 1.0)
    if velocity_vs_plan < 0.6:
        severity = "high" if velocity_vs_plan < 0.4 else "medium"
        pct = int((1 - velocity_vs_plan) * 100)
        bottlenecks.append({
            "type": "velocity_lag",
            "description": f"Sprint velocity is {pct}% behind planned pace",
            "severity": severity,
            "affected_task_ids": [],
        })

    # 5. Deadline risk
    days_remaining = sprint_data.get("days_remaining", 999)
    completion_rate = sprint_data.get("completion_rate", 1.0)
    if days_remaining < 2 and completion_rate < 0.7:
        remaining_pct = int((1 - completion_rate) * 100)
        bottlenecks.append({
            "type": "deadline_risk",
            "description": f"Sprint ends in {days_remaining:.0f} day{'s' if days_remaining != 1 else ''} "
                           f"with {remaining_pct}% of work remaining",
            "severity": "critical",
            "affected_task_ids": [],
        })

    # 6. Scope creep
    capacity_util = sprint_data.get("capacity_utilization", 0)
    if capacity_util > 1.2:
        overcommit_pct = int((capacity_util - 1.0) * 100)
        bottlenecks.append({
            "type": "scope_creep",
            "description": f"Sprint is {overcommit_pct}% over capacity",
            "severity": "high" if capacity_util > 1.4 else "medium",
            "affected_task_ids": [],
        })

    # 7. Workload imbalance
    workload_std = sprint_data.get("workload_std_dev", 0)
    if workload_std > 0.5:
        bottlenecks.append({
            "type": "workload_imbalance",
            "description": "Significant workload imbalance across team members",
            "severity": "medium",
            "affected_task_ids": [],
        })

    # 8. Overdue tasks
    overdue = sprint_data.get("overdue_task_count", 0)
    if overdue > 3:
        bottlenecks.append({
            "type": "overdue_tasks",
            "description": f"{overdue} tasks are past their due date",
            "severity": "high" if overdue > 5 else "medium",
            "affected_task_ids": sprint_data.get("overdue_task_ids", []),
        })

    return bottlenecks


# Test the detector
print("Testing bottleneck detector on sample sprints...")

test_sprints = [
    {"overloaded_member_count": 4, "blocked_task_count": 6, "unassigned_critical_tasks": 2,
     "velocity_vs_plan": 0.3, "days_remaining": 1, "completion_rate": 0.2,
     "capacity_utilization": 1.5, "workload_std_dev": 0.6, "overdue_task_count": 5},
    {"overloaded_member_count": 0, "blocked_task_count": 1, "unassigned_critical_tasks": 0,
     "velocity_vs_plan": 1.1, "days_remaining": 7, "completion_rate": 0.7,
     "capacity_utilization": 0.8, "workload_std_dev": 0.2, "overdue_task_count": 0},
]

for i, sprint in enumerate(test_sprints):
    bns = detect_bottlenecks(sprint)
    print(f"\n  Sprint {i+1}: {len(bns)} bottleneck{'s' if len(bns) != 1 else ''}")
    for bn in bns:
        print(f"    [{bn['severity']:8s}] {bn['type']:25s} — {bn['description']}")


# ============================================================================
# SECTION 6: Recommendation Engine
# ============================================================================
# MARKDOWN: ## 6. Recommendation Engine
#
# Maps bottleneck types to actionable recommendations. Each recommendation
# includes an action, reason, and priority level. The engine considers the
# severity of each bottleneck to prioritize recommendations.

print("\n" + "=" * 60)
print("SECTION 6: Recommendation engine")
print("=" * 60)


def generate_recommendations(bottlenecks: list[dict], sprint_data: dict) -> list[dict]:
    """Generate actionable recommendations from detected bottlenecks.

    Returns a list of recommendation dicts with action, reason, and priority.
    """
    recommendations = []

    for bn in bottlenecks:
        bn_type = bn["type"]
        severity = bn["severity"]

        if bn_type == "member_overload":
            recommendations.append({
                "action": "Redistribute tasks from overloaded members to team members with available capacity",
                "reason": bn["description"],
                "priority": "high" if severity in ("high", "critical") else "medium",
            })
            if severity == "critical":
                recommendations.append({
                    "action": "Consider pulling in additional team members or extending the sprint",
                    "reason": "Multiple team members are critically overloaded",
                    "priority": "high",
                })

        elif bn_type == "blocked_tasks":
            recommendations.append({
                "action": "Schedule an immediate blocker resolution meeting",
                "reason": bn["description"],
                "priority": "high" if severity in ("high", "critical") else "medium",
            })
            if severity == "critical":
                recommendations.append({
                    "action": "Escalate blocking issues to project lead for immediate intervention",
                    "reason": "Blocked tasks are critically impacting sprint progress",
                    "priority": "high",
                })

        elif bn_type == "unassigned_critical":
            recommendations.append({
                "action": "Immediately assign all unassigned critical tasks to available team members",
                "reason": bn["description"],
                "priority": "high",
            })

        elif bn_type == "velocity_lag":
            recommendations.append({
                "action": "Consider reducing sprint scope — move low-priority tasks to the backlog",
                "reason": bn["description"],
                "priority": "high" if severity == "high" else "medium",
            })
            recommendations.append({
                "action": "Hold a focused standup to identify and remove impediments slowing the team",
                "reason": "Team is behind planned velocity",
                "priority": "medium",
            })

        elif bn_type == "deadline_risk":
            recommendations.append({
                "action": "Escalate sprint risk to project lead — sprint is at risk of missing deadline",
                "reason": bn["description"],
                "priority": "high",
            })
            recommendations.append({
                "action": "Identify must-have vs nice-to-have tasks and defer non-essential items",
                "reason": "Limited time remaining requires scope prioritization",
                "priority": "high",
            })

        elif bn_type == "scope_creep":
            over_pct = sprint_data.get("capacity_utilization", 1.0)
            recommendations.append({
                "action": f"Move lowest-priority tasks to backlog to bring sprint within capacity",
                "reason": bn["description"],
                "priority": "medium",
            })

        elif bn_type == "workload_imbalance":
            recommendations.append({
                "action": "Rebalance task assignments to distribute workload more evenly across the team",
                "reason": bn["description"],
                "priority": "medium",
            })

        elif bn_type == "overdue_tasks":
            recommendations.append({
                "action": "Review all overdue tasks and update due dates or reassign to unblock progress",
                "reason": bn["description"],
                "priority": "high" if severity == "high" else "medium",
            })

    # Deduplicate by action
    seen = set()
    unique = []
    for rec in recommendations:
        if rec["action"] not in seen:
            seen.add(rec["action"])
            unique.append(rec)

    # Sort by priority (high first)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    unique.sort(key=lambda r: priority_order.get(r["priority"], 2))

    return unique


# Test the recommendation engine
print("Testing recommendation engine...")
for i, sprint in enumerate(test_sprints):
    bns = detect_bottlenecks(sprint)
    recs = generate_recommendations(bns, sprint)
    print(f"\n  Sprint {i+1}: {len(recs)} recommendation{'s' if len(recs) != 1 else ''}")
    for rec in recs[:5]:
        print(f"    [{rec['priority']:6s}] {rec['action'][:80]}")


# ============================================================================
# SECTION 7: Save & Export
# ============================================================================
# MARKDOWN: ## 7. Save & Export
#
# We save:
# - `optimizer_model.pkl` — XGBoost model
# - `optimizer_scaler.pkl` — StandardScaler
# - `optimizer_explainer.pkl` — SHAP TreeExplainer (for inference-time explanations)
# - `optimizer_label_encoder.pkl` — LabelEncoder for risk level classes
# - `training_metadata.json` — Feature names, metrics, hyperparameters

print("\n" + "=" * 60)
print("SECTION 7: Save & Export")
print("=" * 60)

SAVE_DIR = Path("/content/optimizer-model") if Path("/content").exists() else Path("./optimizer-model")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Save XGBoost model
model_path = SAVE_DIR / "optimizer_model.pkl"
joblib.dump(best_model, model_path)
model_size = model_path.stat().st_size
print(f"Saved XGBoost model: {model_path} ({model_size / 1e6:.2f} MB)")

# Save scaler
scaler_path = SAVE_DIR / "optimizer_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"Saved scaler: {scaler_path} ({scaler_path.stat().st_size / 1e3:.1f} KB)")

# Save SHAP explainer
explainer_path = SAVE_DIR / "optimizer_explainer.pkl"
joblib.dump(explainer, explainer_path)
explainer_size = explainer_path.stat().st_size
print(f"Saved SHAP explainer: {explainer_path} ({explainer_size / 1e6:.2f} MB)")

# Save label encoder
le_path = SAVE_DIR / "optimizer_label_encoder.pkl"
joblib.dump(le, le_path)
print(f"Saved label encoder: {le_path}")

# Save SHAP plot if it was created
for plot_file in ["shap_feature_importance.png", "top_features_bar.png"]:
    if os.path.isfile(plot_file):
        import shutil
        shutil.copy(plot_file, SAVE_DIR / plot_file)
        print(f"Saved plot: {SAVE_DIR / plot_file}")

# Benchmark inference time
print("\nBenchmarking inference time...")
dummy = np.random.randn(1, len(FEATURE_NAMES)).astype(np.float32)
dummy_scaled = scaler.transform(dummy)

# Warm up
for _ in range(20):
    best_model.predict_proba(dummy_scaled)

times = []
for _ in range(200):
    t0 = time.perf_counter()
    best_model.predict_proba(dummy_scaled)
    times.append(time.perf_counter() - t0)

avg_ms = np.mean(times) * 1000
p99_ms = np.percentile(times, 99) * 1000
print(f"  XGBoost predict: avg={avg_ms:.3f}ms, p99={p99_ms:.3f}ms")

# SHAP inference time
shap_times = []
for _ in range(50):
    t0 = time.perf_counter()
    explainer.shap_values(dummy_scaled)
    shap_times.append(time.perf_counter() - t0)

shap_avg_ms = np.mean(shap_times) * 1000
print(f"  SHAP explain:    avg={shap_avg_ms:.2f}ms")
print(f"  Total per sprint: ~{avg_ms + shap_avg_ms:.1f}ms")

target_met = (avg_ms + shap_avg_ms) < 10
if target_met:
    print(f"  TARGET MET: <10ms per sprint prediction")
else:
    print(f"  Note: SHAP adds latency. Consider computing SHAP only on-demand.")

# Save metadata
metadata = {
    "model_type": "XGBClassifier",
    "model_version": "optimizer-v1",
    "best_params": grid_search.best_params_,
    "feature_names": FEATURE_NAMES,
    "feature_count": len(FEATURE_NAMES),
    "risk_levels": RISK_LEVELS,
    "training_examples": len(df),
    "test_accuracy": round(test_acc, 4),
    "test_f1_macro": round(test_f1, 4),
    "cv_f1_macro": round(grid_search.best_score_, 4),
    "top_features_shap": [
        {"name": name, "importance": round(float(imp), 4)}
        for name, imp in shap_importance[:10]
    ],
    "top_features_xgboost": [
        {"name": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])[:10]
    ],
    "bottleneck_types": [
        "member_overload", "blocked_tasks", "unassigned_critical",
        "velocity_lag", "deadline_risk", "scope_creep",
        "workload_imbalance", "overdue_tasks",
    ],
    "inference_time_ms": round(avg_ms, 3),
    "shap_time_ms": round(shap_avg_ms, 2),
    "model_size_mb": round(model_size / 1e6, 2),
}

metadata_path = SAVE_DIR / "training_metadata.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"\nSaved metadata: {metadata_path}")

# List saved files
print(f"\nSaved files in {SAVE_DIR}:")
total_size = 0
for fp in sorted(SAVE_DIR.iterdir()):
    size = fp.stat().st_size
    total_size += size
    label = f"{size / 1e6:.2f} MB" if size > 1e6 else f"{size / 1e3:.1f} KB"
    print(f"  {fp.name}: {label}")
print(f"  Total: {total_size / 1e6:.2f} MB")

# Zip for download
import shutil
zip_path = shutil.make_archive(
    base_name=str(SAVE_DIR),
    format="zip",
    root_dir=str(SAVE_DIR.parent),
    base_dir=SAVE_DIR.name,
)
zip_size = os.path.getsize(zip_path)
print(f"\nZip archive: {zip_path} ({zip_size / 1e6:.2f} MB)")

# Download in Colab
try:
    from google.colab import files
    print("Downloading zip to your browser...")
    files.download(zip_path)
except ImportError:
    print(f"Not in Colab — zip saved locally at: {zip_path}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\nModel: XGBClassifier ({grid_search.best_params_})")
print(f"Accuracy: {test_acc:.4f}")
print(f"Macro F1: {test_f1:.4f}")
print(f"Inference: {avg_ms:.3f}ms per sprint")
print(f"\nTop 5 most important features:")
for name, imp in shap_importance[:5]:
    print(f"  {name}: {imp:.4f}")
print(f"\nTo deploy in AI server:")
print(f"  1. Unzip optimizer-model.zip")
print(f"  2. Copy contents to ai-server/model_weights/optimizer/")
print(f"  3. Restart the AI server")
print(f"  4. /health should show \"optimizer\": true")
