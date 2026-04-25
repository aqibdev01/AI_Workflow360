#!/usr/bin/env python3
"""
Combined training script for Workflow360 AI modules.
Trains Assigner (M10) and Optimizer (M11) with exact versions
matching HF Space requirements.txt to avoid pkl version mismatch.

Run from the ai-server directory:
    python notebooks/run_training.py
"""

import subprocess
import sys
import os

# ─── Step 0: Pin exact versions matching requirements.txt ───────────────────
print("=" * 60)
print("Installing exact package versions matching HF Space...")
print("=" * 60)

packages = [
    "scikit-learn==1.5.0",
    "sentence-transformers==3.0.0",
    "xgboost==2.0.3",
    "shap==0.45.1",
    "joblib==1.4.2",
    "numpy==1.26.4",
    "pandas>=2.2.0",
]

for pkg in packages:
    print(f"  pip install {pkg}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        stdout=subprocess.DEVNULL,
    )

print("Packages ready.\n")

# ─── Imports ─────────────────────────────────────────────────────────────────
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import xgboost as xgb
import shap

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Output directories — relative to project root
ROOT = Path(__file__).parent.parent
ASSIGNER_OUT = ROOT / "model_weights" / "assigner"
OPTIMIZER_OUT = ROOT / "model_weights" / "optimizer"
ASSIGNER_OUT.mkdir(parents=True, exist_ok=True)
OPTIMIZER_OUT.mkdir(parents=True, exist_ok=True)

print(f"Model weights will be saved to:")
print(f"  Assigner  -> {ASSIGNER_OUT}")
print(f"  Optimizer -> {OPTIMIZER_OUT}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: ASSIGNER (M10) — Sentence-BERT + GradientBoosting
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("MODULE 1: Training Task Assigner (M10)")
print("=" * 60)

# ── Skill & task pools ────────────────────────────────────────────────────────
SKILL_POOLS = {
    "frontend": ["React", "TypeScript", "CSS", "Tailwind", "Next.js", "HTML", "JavaScript", "Vue", "Angular", "Figma"],
    "backend": ["Python", "Node.js", "PostgreSQL", "REST API", "GraphQL", "FastAPI", "Django", "Express", "Redis", "SQL"],
    "devops": ["Docker", "Kubernetes", "CI/CD", "AWS", "Linux", "Terraform", "GitHub Actions", "Nginx", "Monitoring", "Shell"],
    "testing": ["Jest", "Playwright", "Cypress", "Unit Testing", "Integration Testing", "E2E", "TDD", "QA", "Selenium", "Load Testing"],
    "design": ["Figma", "UI Design", "UX Research", "Wireframing", "Prototyping", "Design Systems", "Accessibility", "Sketch"],
    "documentation": ["Technical Writing", "API Docs", "Markdown", "Confluence", "Swagger", "Documentation", "Tutorials"],
}
ALL_SKILLS = list({s for pool in SKILL_POOLS.values() for s in pool})

TASK_TEMPLATES = {
    "frontend": [
        {"title": "Build dashboard component", "description": "Create a responsive dashboard page with charts and stats", "tags": ["React", "CSS", "Tailwind"], "priority": "high", "story_points": 5},
        {"title": "Implement kanban drag-and-drop", "description": "Add DnD Kit drag-and-drop for kanban board columns", "tags": ["React", "TypeScript", "JavaScript"], "priority": "high", "story_points": 8},
        {"title": "Add form validation", "description": "Implement Zod validation schemas for all user-facing forms", "tags": ["TypeScript", "React", "JavaScript"], "priority": "medium", "story_points": 3},
        {"title": "Create notification dropdown", "description": "Build notification center dropdown with real-time updates", "tags": ["React", "TypeScript", "CSS"], "priority": "medium", "story_points": 5},
        {"title": "Implement dark mode", "description": "Add dark mode support across all components using Tailwind", "tags": ["CSS", "Tailwind", "React"], "priority": "low", "story_points": 3},
    ],
    "backend": [
        {"title": "Build task CRUD API", "description": "Create REST endpoints for task creation, reading, updating and deleting", "tags": ["REST API", "Node.js", "PostgreSQL", "SQL"], "priority": "high", "story_points": 8},
        {"title": "Implement authentication", "description": "Set up email/password auth with signup, login, and password reset flows", "tags": ["Node.js", "Python", "REST API"], "priority": "high", "story_points": 8},
        {"title": "Optimize database queries", "description": "Add indexes and optimize slow queries for task listing and sprint stats", "tags": ["PostgreSQL", "SQL"], "priority": "medium", "story_points": 3},
        {"title": "Implement caching layer", "description": "Add Redis caching for frequently accessed project and member data", "tags": ["Redis", "Node.js", "Python"], "priority": "medium", "story_points": 5},
        {"title": "Build AI server integration", "description": "Create API routes that proxy requests to the FastAPI AI server", "tags": ["FastAPI", "Python", "REST API"], "priority": "high", "story_points": 5},
    ],
    "devops": [
        {"title": "Set up CI/CD pipeline", "description": "Configure GitHub Actions with lint, test, build and deploy stages", "tags": ["CI/CD", "GitHub Actions", "Docker"], "priority": "high", "story_points": 5},
        {"title": "Containerize AI server", "description": "Create Docker setup for the FastAPI AI server with multi-stage builds", "tags": ["Docker", "Linux", "Python"], "priority": "medium", "story_points": 5},
        {"title": "Configure monitoring", "description": "Set up application monitoring, structured logging and alerting", "tags": ["Monitoring", "Linux", "AWS"], "priority": "medium", "story_points": 5},
        {"title": "Set up SSL certificates", "description": "Configure SSL/TLS certificates and HTTPS for all services", "tags": ["Nginx", "Linux", "AWS"], "priority": "high", "story_points": 2},
    ],
    "testing": [
        {"title": "Set up testing framework", "description": "Configure Jest and React Testing Library with mocks and coverage", "tags": ["Jest", "Unit Testing", "TDD"], "priority": "high", "story_points": 3},
        {"title": "Write API integration tests", "description": "Create integration tests for all task CRUD endpoints", "tags": ["Integration Testing", "Jest", "Unit Testing"], "priority": "high", "story_points": 5},
        {"title": "Write E2E tests", "description": "Create Playwright end-to-end tests for critical user flows", "tags": ["Playwright", "E2E", "Cypress"], "priority": "medium", "story_points": 8},
        {"title": "Create test data factories", "description": "Build reusable test fixtures and factory functions for all models", "tags": ["Jest", "Unit Testing", "Integration Testing"], "priority": "medium", "story_points": 3},
    ],
    "design": [
        {"title": "Design UI kit", "description": "Create comprehensive UI kit with reusable component designs", "tags": ["Figma", "UI Design", "Design Systems"], "priority": "medium", "story_points": 5},
        {"title": "Create AI feature wireframes", "description": "Design wireframes for AI-powered decomposition, assignment, and sprint risk panels", "tags": ["Wireframing", "UX Research", "Figma", "UI Design"], "priority": "high", "story_points": 5},
        {"title": "Design mobile layout", "description": "Adapt the PM interface for mobile with touch-friendly interactions", "tags": ["UI Design", "Figma", "Prototyping", "Accessibility"], "priority": "low", "story_points": 5},
    ],
    "documentation": [
        {"title": "Write API documentation", "description": "Create comprehensive API docs with request/response examples", "tags": ["API Docs", "Swagger", "Technical Writing", "Documentation"], "priority": "medium", "story_points": 5},
        {"title": "Create setup guide", "description": "Write getting-started guide for new developers covering all services", "tags": ["Technical Writing", "Markdown", "Documentation"], "priority": "high", "story_points": 3},
        {"title": "Write architecture docs", "description": "Document key architectural decisions and technology choices", "tags": ["Technical Writing", "Documentation", "Confluence"], "priority": "low", "story_points": 3},
    ],
}

MEMBER_TEMPLATES = [
    {"name": "Frontend Lead",    "role": "lead",        "skills": ["React", "TypeScript", "CSS", "Tailwind", "Next.js", "JavaScript", "HTML"]},
    {"name": "Backend Dev",      "role": "contributor", "skills": ["Python", "PostgreSQL", "FastAPI", "REST API", "SQL", "Redis"]},
    {"name": "Full Stack Dev 1", "role": "contributor", "skills": ["React", "Node.js", "TypeScript", "PostgreSQL", "REST API", "JavaScript"]},
    {"name": "Full Stack Dev 2", "role": "contributor", "skills": ["React", "Python", "TypeScript", "Django", "CSS", "SQL"]},
    {"name": "DevOps Engineer",  "role": "contributor", "skills": ["Docker", "Kubernetes", "CI/CD", "AWS", "Linux", "Terraform", "GitHub Actions", "Nginx", "Monitoring", "Shell"]},
    {"name": "QA Engineer",      "role": "contributor", "skills": ["Jest", "Playwright", "Cypress", "Unit Testing", "Integration Testing", "E2E", "QA", "Selenium", "Load Testing"]},
    {"name": "UI Designer",      "role": "contributor", "skills": ["Figma", "UI Design", "UX Research", "Wireframing", "Prototyping", "Design Systems", "Accessibility", "CSS"]},
    {"name": "Tech Writer",      "role": "contributor", "skills": ["Technical Writing", "API Docs", "Markdown", "Confluence", "Swagger", "Documentation", "Tutorials"]},
    {"name": "Junior Dev 1",     "role": "contributor", "skills": ["JavaScript", "HTML", "CSS", "React"]},
    {"name": "Junior Dev 2",     "role": "contributor", "skills": ["Python", "SQL", "Node.js"]},
    {"name": "Team Lead",        "role": "lead",        "skills": ["React", "Python", "Node.js", "PostgreSQL", "Docker", "CI/CD", "TypeScript", "REST API", "AWS"]},
    {"name": "Project Owner",    "role": "owner",       "skills": ["Technical Writing", "UX Research", "REST API"]},
]

ROLE_CATEGORY_FIT = {
    "Frontend Lead":    {"frontend": 1.0, "backend": 0.3, "testing": 0.5, "design": 0.6, "devops": 0.1, "documentation": 0.3},
    "Backend Dev":      {"frontend": 0.1, "backend": 1.0, "testing": 0.5, "design": 0.0, "devops": 0.4, "documentation": 0.3},
    "Full Stack Dev 1": {"frontend": 0.8, "backend": 0.8, "testing": 0.5, "design": 0.2, "devops": 0.2, "documentation": 0.3},
    "Full Stack Dev 2": {"frontend": 0.7, "backend": 0.7, "testing": 0.4, "design": 0.3, "devops": 0.2, "documentation": 0.3},
    "DevOps Engineer":  {"frontend": 0.0, "backend": 0.3, "testing": 0.3, "design": 0.0, "devops": 1.0, "documentation": 0.3},
    "QA Engineer":      {"frontend": 0.2, "backend": 0.2, "testing": 1.0, "design": 0.1, "devops": 0.2, "documentation": 0.3},
    "UI Designer":      {"frontend": 0.5, "backend": 0.0, "testing": 0.1, "design": 1.0, "devops": 0.0, "documentation": 0.3},
    "Tech Writer":      {"frontend": 0.1, "backend": 0.1, "testing": 0.1, "design": 0.2, "devops": 0.1, "documentation": 1.0},
    "Junior Dev 1":     {"frontend": 0.5, "backend": 0.2, "testing": 0.3, "design": 0.2, "devops": 0.1, "documentation": 0.2},
    "Junior Dev 2":     {"frontend": 0.2, "backend": 0.5, "testing": 0.3, "design": 0.0, "devops": 0.1, "documentation": 0.2},
    "Team Lead":        {"frontend": 0.7, "backend": 0.7, "testing": 0.4, "design": 0.3, "devops": 0.5, "documentation": 0.4},
    "Project Owner":    {"frontend": 0.2, "backend": 0.2, "testing": 0.2, "design": 0.4, "devops": 0.2, "documentation": 0.6},
}

MAX_CAPACITY = 40

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

def build_task_text(task):
    parts = [task["title"]]
    if task.get("description"):
        parts.append(task["description"])
    if task.get("tags"):
        parts.append("Tags: " + ", ".join(task["tags"]))
    return " ".join(parts)

def build_member_text(member):
    return f"Skills: {', '.join(member['skills'])}. Role: {member['role']}"

def ground_truth_score(task, member, category, all_members):
    task_tags = {t.lower() for t in task["tags"]}
    member_skills = {s.lower() for s in member["skills"]}
    skill_match = jaccard(task_tags, member_skills)
    max_wl = max(m["current_workload"] for m in all_members) or 1
    workload = 1.0 - (member["current_workload"] / (max_wl + 1))
    role_fit = ROLE_CATEGORY_FIT.get(member["template_name"], {}).get(category, 0.3)
    max_comp = max(m["completed_last_30d"] for m in all_members) or 1
    performance = member["completed_last_30d"] / max_comp
    return 0.40 * skill_match + 0.25 * workload + 0.20 * role_fit + 0.15 * performance

def generate_examples(n=5000):
    examples = []
    for _ in range(n):
        category = random.choice(list(TASK_TEMPLATES.keys()))
        task = random.choice(TASK_TEMPLATES[category]).copy()
        task["story_points"] = random.choice([1, 2, 3, 5, 8])
        task["priority"] = random.choice(["low", "medium", "high", "urgent"])
        team_size = random.randint(3, 8)
        team = random.sample(MEMBER_TEMPLATES, min(team_size, len(MEMBER_TEMPLATES)))
        members = []
        for tpl in team:
            m = {
                "template_name": tpl["name"],
                "user_id": f"user-{tpl['name'].lower().replace(' ', '-')}",
                "role": tpl["role"],
                "skills": tpl["skills"].copy(),
                "current_workload": random.randint(0, 40),
                "completed_last_30d": random.randint(2, 25),
            }
            if random.random() > 0.7 and len(m["skills"]) > 2:
                m["skills"].pop(random.randint(0, len(m["skills"]) - 1))
            if random.random() > 0.8:
                m["skills"] = list(set(m["skills"] + [random.choice(ALL_SKILLS)]))
            members.append(m)
        scores = [ground_truth_score(task, m, category, members) for m in members]
        best_idx = int(np.argmax(scores))
        if random.random() < 0.10 and len(scores) > 1:
            best_idx = int(np.argsort(scores)[-2])
        examples.append({"task": task, "category": category, "members": members, "correct_assignee_idx": best_idx})
    return examples

print("Generating 5000 training examples...")
all_examples = generate_examples(5000)
print(f"Generated {len(all_examples)} examples")

print("Loading Sentence-BERT (all-MiniLM-L6-v2)...")
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"SBERT loaded. Embedding dim: {sbert.get_sentence_embedding_dimension()}")

print("Computing embeddings and building feature matrix...")
unique_task_texts = list({build_task_text(e["task"]) for e in all_examples})
unique_member_texts = list({build_member_text(m) for e in all_examples for m in e["members"]})

task_emb_map = dict(zip(unique_task_texts, sbert.encode(unique_task_texts, batch_size=128, show_progress_bar=False)))
member_emb_map = dict(zip(unique_member_texts, sbert.encode(unique_member_texts, batch_size=128, show_progress_bar=False)))

X_rows, y_rows = [], []
for ex in all_examples:
    task = ex["task"]
    task_emb = task_emb_map[build_task_text(task)]
    for i, m in enumerate(ex["members"]):
        m_emb = member_emb_map[build_member_text(m)]
        task_tags = {t.lower() for t in task["tags"]}
        m_skills = {s.lower() for s in m["skills"]}
        skill_match = jaccard(task_tags, m_skills)
        sem_sim = float(cos_sim(task_emb.reshape(1, -1), m_emb.reshape(1, -1))[0, 0])
        workload = max(0.0, 1.0 - (m["current_workload"] / MAX_CAPACITY))
        role = ROLE_CATEGORY_FIT.get(m.get("template_name", ""), {}).get(ex["category"], 0.3)
        max_comp = max(mm["completed_last_30d"] for mm in ex["members"]) or 1
        perf = m["completed_last_30d"] / max_comp
        wl = m["current_workload"]
        avail = 1.0 if wl < 15 else 0.5 if wl < 30 else 0.0
        X_rows.append([skill_match, sem_sim, workload, role, perf, avail])
        y_rows.append(1 if i == ex["correct_assignee_idx"] else 0)

X = np.array(X_rows, dtype=np.float32)
y = np.array(y_rows, dtype=np.int32)
print(f"Feature matrix: {X.shape}, positives: {y.sum()} ({y.mean()*100:.1f}%)")

scaler_a = StandardScaler()
X_scaled = scaler_a.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=SEED, stratify=y)

print("\nTraining GradientBoosting classifier...")
model_a = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=SEED)
model_a.fit(X_train, y_train)
f1_a = f1_score(y_test, model_a.predict(X_test))

print("Training RandomForest classifier...")
model_b = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
model_b.fit(X_train, y_train)
f1_b = f1_score(y_test, model_b.predict(X_test))

best_model_a = model_a if f1_a >= f1_b else model_b
best_name_a = "GradientBoosting" if f1_a >= f1_b else "RandomForest"
print(f"Winner: {best_name_a} (F1: {max(f1_a, f1_b):.4f})")

print("Calibrating probabilities (CalibratedClassifierCV)...")
calibrated = CalibratedClassifierCV(best_model_a, method="sigmoid", cv=3)
calibrated.fit(X_train, y_train)
y_proba = calibrated.predict_proba(X_test)[:, 1]
y_pred_cal = (y_proba >= 0.5).astype(int)
cal_acc = accuracy_score(y_test, y_pred_cal)
cal_f1 = f1_score(y_test, y_pred_cal)
print(f"Calibrated — accuracy: {cal_acc:.4f}, F1: {cal_f1:.4f}")

FEATURE_NAMES_A = ["skill_match_score", "semantic_similarity", "workload_score", "role_match_score", "performance_score", "availability_score"]
importances_a = best_model_a.feature_importances_

print(f"\nSaving assigner model to {ASSIGNER_OUT}...")
joblib.dump(calibrated, ASSIGNER_OUT / "assigner_model.pkl")
joblib.dump(scaler_a, ASSIGNER_OUT / "assigner_scaler.pkl")
metadata_a = {
    "model_type": best_name_a,
    "model_version": "assigner-v1",
    "calibrated": True,
    "sbert_model": "sentence-transformers/all-MiniLM-L6-v2",
    "feature_names": FEATURE_NAMES_A,
    "feature_count": len(FEATURE_NAMES_A),
    "training_examples": len(all_examples),
    "test_accuracy": round(cal_acc, 4),
    "test_f1": round(cal_f1, 4),
    "sklearn_version": "1.5.0",
    "feature_importances": {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES_A, importances_a)},
}
with open(ASSIGNER_OUT / "training_metadata.json", "w") as f:
    json.dump(metadata_a, f, indent=2)

print("Assigner files saved:")
for p in ASSIGNER_OUT.iterdir():
    print(f"  {p.name} ({p.stat().st_size / 1024:.1f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: OPTIMIZER (M11) — XGBoost + SHAP
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("MODULE 2: Training Sprint Optimizer (M11)")
print("=" * 60)

FEATURE_NAMES_O = [
    "days_remaining", "completion_rate", "story_points_completed",
    "story_points_remaining", "capacity_utilization", "past_velocity_avg",
    "blocked_task_count", "critical_task_count", "unassigned_task_count",
    "overdue_task_count", "tasks_without_estimates",
    "max_member_workload", "workload_std_dev", "avg_member_utilization",
    "overloaded_member_count", "velocity_trend", "velocity_vs_plan",
    "scope_change_ratio",
]
RISK_LEVELS = ["low", "medium", "high", "critical"]


def label_sprint(f):
    if f["completion_rate"] < 0.2 and f["days_remaining"] < 3:
        return "critical"
    if f["overloaded_member_count"] > 3:
        return "critical"
    if f["blocked_task_count"] > 6 and f["days_remaining"] < 4:
        return "critical"
    if f["completion_rate"] < 0.4 and f["days_remaining"] < 5:
        return "high"
    if f["blocked_task_count"] > 4:
        return "high"
    if f["overloaded_member_count"] > 2 and f["velocity_vs_plan"] < 0.6:
        return "high"
    if f["critical_task_count"] > 3 and f["unassigned_task_count"] > 2:
        return "high"
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


def generate_sprint(bias=None):
    sprint_length = random.choice([7, 10, 14])
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
    else:
        days_remaining = random.uniform(5.0, 12.0)
        completion_rate = random.uniform(0.5, 0.95)
        overloaded_count = random.randint(0, 1)
        blocked = random.randint(0, 2)

    sp_completed = round(total_sp * completion_rate, 1)
    sp_remaining = round(total_sp - sp_completed, 1)
    past_velocities = [random.uniform(15, 60) for _ in range(3)]
    member_workloads = [random.uniform(5, 30) for _ in range(team_size)]
    mean_wl = np.mean(member_workloads)

    elapsed = (sprint_length - days_remaining) / sprint_length if sprint_length > 0 else 0.9
    vvp = min(2.0, completion_rate / max(elapsed, 0.01))

    return {
        "days_remaining": round(days_remaining, 1),
        "completion_rate": round(completion_rate, 3),
        "story_points_completed": sp_completed,
        "story_points_remaining": sp_remaining,
        "capacity_utilization": round(total_sp / max(capacity, 1), 2),
        "past_velocity_avg": round(np.mean(past_velocities), 1),
        "blocked_task_count": blocked,
        "critical_task_count": random.randint(0, max(1, total_tasks // 4)),
        "unassigned_task_count": random.randint(0, max(1, total_tasks // 3)),
        "overdue_task_count": random.randint(0, max(1, int(total_tasks * (1 - completion_rate) * 0.5))),
        "tasks_without_estimates": random.randint(0, max(1, total_tasks // 5)),
        "max_member_workload": round(max(member_workloads), 1),
        "workload_std_dev": round(float(np.std(member_workloads) / max(mean_wl, 1)), 3),
        "avg_member_utilization": round(float(mean_wl) / max(capacity / team_size, 1), 3),
        "overloaded_member_count": overloaded_count,
        "velocity_trend": round((past_velocities[-1] - past_velocities[0]) / max(past_velocities[0], 1), 3),
        "velocity_vs_plan": round(vvp, 3),
        "scope_change_ratio": round(random.uniform(-0.1, 0.3), 3),
    }


def generate_optimizer_dataset(n=8000):
    per_class = n // 4
    examples = []
    for risk_class in RISK_LEVELS:
        count, attempts = 0, 0
        while count < per_class and attempts < per_class * 10:
            f = generate_sprint(bias=risk_class)
            label = label_sprint(f)
            if label == risk_class or attempts > per_class * 5:
                f["risk_level"] = label
                examples.append(f)
                count += 1
            attempts += 1
    random.shuffle(examples)
    df = pd.DataFrame(examples)
    # 5% label noise
    flip_idx = random.sample(range(len(df)), int(0.05 * len(df)))
    for idx in flip_idx:
        cur = df.at[idx, "risk_level"]
        df.at[idx, "risk_level"] = "medium" if cur == "low" else "high" if cur == "critical" else random.choice([r for r in RISK_LEVELS if r != cur])
    return df


print("Generating 8000 sprint examples...")
df_opt = generate_optimizer_dataset(8000)
print(f"Dataset shape: {df_opt.shape}")
print(f"Risk distribution:\n{df_opt['risk_level'].value_counts().sort_index().to_string()}")

X_o = df_opt[FEATURE_NAMES_O].values.astype(np.float32)
le = LabelEncoder()
y_o = le.fit_transform(df_opt["risk_level"])
class_names = le.classes_.tolist()
print(f"\nClasses: {class_names}")

scaler_o = StandardScaler()
X_o_scaled = scaler_o.fit_transform(X_o)
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_o_scaled, y_o, test_size=0.2, random_state=SEED, stratify=y_o)

print("\nRunning XGBoost GridSearchCV (this takes ~5-10 min on CPU)...")
param_grid_o = {
    "n_estimators": [100, 200],
    "max_depth": [3, 4],
    "learning_rate": [0.1, 0.2],
}
base_xgb = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(class_names),
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=SEED,
    verbosity=0,
)
cv_o = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
t0 = time.time()
gs = GridSearchCV(base_xgb, param_grid_o, cv=cv_o, scoring="f1_macro", n_jobs=-1, verbose=1, refit=True)
gs.fit(X_train_o, y_train_o)
print(f"Grid search done in {time.time()-t0:.1f}s. Best params: {gs.best_params_}")

best_xgb = gs.best_estimator_
y_pred_o = best_xgb.predict(X_test_o)
test_acc_o = accuracy_score(y_test_o, y_pred_o)
test_f1_o = f1_score(y_test_o, y_pred_o, average="macro")
print(f"Test accuracy: {test_acc_o:.4f}, Macro F1: {test_f1_o:.4f}")
print(classification_report(y_test_o, y_pred_o, target_names=class_names))

print("\nComputing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(best_xgb)
shap_sample = X_test_o[:300]
shap_values = explainer.shap_values(shap_sample)
print("SHAP done.")

print(f"\nSaving optimizer model to {OPTIMIZER_OUT}...")
joblib.dump(best_xgb, OPTIMIZER_OUT / "optimizer_model.pkl")
joblib.dump(scaler_o, OPTIMIZER_OUT / "optimizer_scaler.pkl")
joblib.dump(explainer, OPTIMIZER_OUT / "optimizer_explainer.pkl")
joblib.dump(le, OPTIMIZER_OUT / "optimizer_label_encoder.pkl")

metadata_o = {
    "model_version": "optimizer-v1",
    "feature_names": FEATURE_NAMES_O,
    "class_names": class_names,
    "best_params": gs.best_params_,
    "test_accuracy": round(test_acc_o, 4),
    "test_f1_macro": round(test_f1_o, 4),
    "sklearn_version": "1.5.0",
}
with open(OPTIMIZER_OUT / "training_metadata.json", "w") as f:
    json.dump(metadata_o, f, indent=2)

print("Optimizer files saved:")
for p in OPTIMIZER_OUT.iterdir():
    print(f"  {p.name} ({p.stat().st_size / 1024:.1f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\nAssigner model: {ASSIGNER_OUT}")
print(f"Optimizer model: {OPTIMIZER_OUT}")
print("\nNext step: push these files to HuggingFace Spaces git repo")
print("  cd <hf-space-local-clone>")
print("  cp -r model_weights/ .")
print("  git add model_weights/")
print("  git commit -m 'Add trained model weights'")
print("  git push")
