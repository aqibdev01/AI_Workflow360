#!/usr/bin/env python3
"""
=============================================================================
Task Assigner Model Training — Sentence-BERT + Scikit-learn (M10)
=============================================================================
Workflow360 FYP — AI Module Training

This script trains a skill-matching + workload scoring system for intelligent
task assignment. NOT a language model — uses Sentence-BERT embeddings as
features fed into a Gradient Boosting classifier.

Trainable on CPU in minutes. No GPU required.

Architecture:
  1. Sentence-BERT (all-MiniLM-L6-v2, 22MB) computes semantic similarity
     between task descriptions and member skill profiles.
  2. Hand-crafted features: skill overlap, workload, role match, performance.
  3. GradientBoosting or RandomForest classifier predicts the best assignee.
  4. CalibratedClassifierCV provides proper confidence probabilities.

Usage:
  python train_assigner.py          # local run
  # Or run cell-by-cell in Google Colab
=============================================================================
"""

# ============================================================================
# MARKDOWN: # Task Assigner Model Training
#
# Trains a **Sentence-BERT + Scikit-learn** classifier for intelligent task
# assignment in Workflow360.
#
# **Input:** Task metadata + team member profiles
# **Output:** Best assignee prediction with confidence score
# **Hardware:** CPU only (~5-10 minutes training)
# **Target:** >85% accuracy on synthetic data, <50ms inference per prediction
# ============================================================================


# ============================================================================
# SECTION 1: Setup & Dependencies
# ============================================================================
# MARKDOWN: ## 1. Setup & Dependencies
# Install lightweight ML packages. Sentence-BERT uses the `all-MiniLM-L6-v2`
# model (22MB) which runs efficiently on CPU.

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

install("sentence-transformers>=3.0.0")
install("scikit-learn>=1.5.0")
install("pandas>=2.2.0")
install("numpy>=1.26.0")
install("joblib>=1.4.0")

import os
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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
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


# ============================================================================
# SECTION 2: Synthetic Dataset Generation
# ============================================================================
# MARKDOWN: ## 2. Synthetic Dataset Generation
#
# We generate 5000 training examples of `(task, members, correct_assignee)` triples.
#
# **Ground truth scoring rules** (this IS the training signal):
# | Dimension | Weight | Logic |
# |-----------|--------|-------|
# | Skill match | 0.40 | Jaccard overlap of task tags and member skills |
# | Workload | 0.25 | Inverse of current story points (lower = better) |
# | Role appropriateness | 0.20 | Does the member's role fit the task category? |
# | Past performance | 0.15 | Completed tasks in last 30 days (higher = better) |

print("\n" + "="*60)
print("SECTION 2: Generating synthetic dataset")
print("="*60)

# --- Skill pools by category ---
SKILL_POOLS = {
    "frontend": ["React", "TypeScript", "CSS", "Tailwind", "Next.js", "HTML", "JavaScript", "Vue", "Angular", "Figma"],
    "backend": ["Python", "Node.js", "PostgreSQL", "REST API", "GraphQL", "FastAPI", "Django", "Express", "Redis", "SQL"],
    "devops": ["Docker", "Kubernetes", "CI/CD", "AWS", "Linux", "Terraform", "GitHub Actions", "Nginx", "Monitoring", "Shell"],
    "testing": ["Jest", "Playwright", "Cypress", "Unit Testing", "Integration Testing", "E2E", "TDD", "QA", "Selenium", "Load Testing"],
    "design": ["Figma", "UI Design", "UX Research", "Wireframing", "Prototyping", "Design Systems", "Accessibility", "Sketch"],
    "documentation": ["Technical Writing", "API Docs", "Markdown", "Confluence", "Swagger", "Documentation", "Tutorials"],
}

ALL_SKILLS = list({s for pool in SKILL_POOLS.values() for s in pool})

# --- Task templates ---
TASK_TEMPLATES = {
    "frontend": [
        {"title": "Build dashboard component", "description": "Create a responsive dashboard page with charts and stats cards using React and Tailwind CSS", "tags": ["React", "CSS", "Tailwind", "UI Design"], "priority": "high", "story_points": 5},
        {"title": "Implement drag-and-drop kanban board", "description": "Add DnD Kit based drag-and-drop for the kanban board columns", "tags": ["React", "TypeScript", "JavaScript"], "priority": "high", "story_points": 8},
        {"title": "Add form validation", "description": "Implement Zod validation schemas for all user-facing forms", "tags": ["TypeScript", "React", "JavaScript"], "priority": "medium", "story_points": 3},
        {"title": "Create notification dropdown", "description": "Build the notification center dropdown with real-time updates", "tags": ["React", "TypeScript", "CSS"], "priority": "medium", "story_points": 5},
        {"title": "Design settings page", "description": "Create multi-tab settings page with profile and preferences", "tags": ["React", "Tailwind", "CSS", "UI Design"], "priority": "low", "story_points": 3},
        {"title": "Implement dark mode", "description": "Add dark mode support across all components using Tailwind", "tags": ["CSS", "Tailwind", "React"], "priority": "low", "story_points": 3},
        {"title": "Build search and filters", "description": "Add search bar with auto-complete and advanced filter dropdowns", "tags": ["React", "TypeScript", "JavaScript"], "priority": "medium", "story_points": 5},
        {"title": "Create analytics charts", "description": "Build reusable chart components with Recharts for project analytics", "tags": ["React", "JavaScript", "TypeScript"], "priority": "medium", "story_points": 5},
        {"title": "Fix mobile responsive layout", "description": "Adapt project management views for mobile breakpoints", "tags": ["CSS", "Tailwind", "HTML", "React"], "priority": "high", "story_points": 3},
        {"title": "Implement onboarding wizard", "description": "Build multi-step onboarding flow for new users with progress tracking", "tags": ["React", "TypeScript", "Next.js"], "priority": "high", "story_points": 8},
    ],
    "backend": [
        {"title": "Build task CRUD API", "description": "Create REST endpoints for task creation, reading, updating and deleting with validation", "tags": ["REST API", "Node.js", "PostgreSQL", "SQL"], "priority": "high", "story_points": 8},
        {"title": "Implement authentication", "description": "Set up email/password auth with signup, login, and password reset flows", "tags": ["Node.js", "Python", "REST API"], "priority": "high", "story_points": 8},
        {"title": "Create sprint API", "description": "Build endpoints for sprint management with state machine transitions", "tags": ["REST API", "PostgreSQL", "Node.js", "SQL"], "priority": "high", "story_points": 5},
        {"title": "Set up real-time notifications", "description": "Implement server-side notification system with database triggers", "tags": ["PostgreSQL", "Node.js", "SQL", "Redis"], "priority": "medium", "story_points": 5},
        {"title": "Build file upload service", "description": "Create file upload endpoints with type validation and version tracking", "tags": ["Node.js", "REST API", "Python"], "priority": "medium", "story_points": 5},
        {"title": "Optimize database queries", "description": "Add indexes and optimize slow queries for task listing and sprint stats", "tags": ["PostgreSQL", "SQL"], "priority": "medium", "story_points": 3},
        {"title": "Implement rate limiting", "description": "Add rate limiting middleware for authentication and API endpoints", "tags": ["Node.js", "Express", "Redis"], "priority": "high", "story_points": 3},
        {"title": "Build AI server integration", "description": "Create API routes that proxy requests to the FastAPI AI server", "tags": ["FastAPI", "Python", "REST API"], "priority": "high", "story_points": 5},
        {"title": "Add webhook endpoints", "description": "Create webhook handlers for external service integrations", "tags": ["REST API", "Node.js", "Express"], "priority": "low", "story_points": 3},
        {"title": "Implement caching layer", "description": "Add Redis caching for frequently accessed project and member data", "tags": ["Redis", "Node.js", "Python"], "priority": "medium", "story_points": 5},
    ],
    "devops": [
        {"title": "Set up CI/CD pipeline", "description": "Configure GitHub Actions with lint, test, build and deploy stages", "tags": ["CI/CD", "GitHub Actions", "Docker"], "priority": "high", "story_points": 5},
        {"title": "Containerize AI server", "description": "Create Docker setup for the FastAPI AI server with multi-stage builds", "tags": ["Docker", "Linux", "Python"], "priority": "medium", "story_points": 5},
        {"title": "Configure monitoring", "description": "Set up application monitoring, structured logging and alerting", "tags": ["Monitoring", "Linux", "AWS"], "priority": "medium", "story_points": 5},
        {"title": "Set up staging environment", "description": "Create staging deployment that mirrors production for testing", "tags": ["AWS", "Docker", "Kubernetes", "Terraform"], "priority": "medium", "story_points": 5},
        {"title": "Implement auto-scaling", "description": "Configure auto-scaling rules for the AI server based on request load", "tags": ["Kubernetes", "AWS", "Docker", "Monitoring"], "priority": "low", "story_points": 8},
        {"title": "Set up SSL certificates", "description": "Configure SSL/TLS certificates and HTTPS for all services", "tags": ["Nginx", "Linux", "AWS"], "priority": "high", "story_points": 2},
        {"title": "Create backup strategy", "description": "Set up automated database backups with point-in-time recovery", "tags": ["AWS", "Linux", "Shell"], "priority": "high", "story_points": 3},
        {"title": "Optimize Docker images", "description": "Reduce Docker image sizes with multi-stage builds and alpine bases", "tags": ["Docker", "Linux", "Shell"], "priority": "low", "story_points": 3},
    ],
    "testing": [
        {"title": "Set up testing framework", "description": "Configure Jest and React Testing Library with mocks and coverage", "tags": ["Jest", "Unit Testing", "TDD"], "priority": "high", "story_points": 3},
        {"title": "Write API integration tests", "description": "Create integration tests for all task CRUD endpoints", "tags": ["Integration Testing", "Jest", "Unit Testing"], "priority": "high", "story_points": 5},
        {"title": "Write E2E tests", "description": "Create Playwright end-to-end tests for critical user flows", "tags": ["Playwright", "E2E", "Cypress"], "priority": "medium", "story_points": 8},
        {"title": "Add unit tests for utils", "description": "Write unit tests for shared utility functions and validators", "tags": ["Jest", "Unit Testing", "TDD"], "priority": "low", "story_points": 3},
        {"title": "Set up load testing", "description": "Configure load testing for API endpoints using k6 or similar", "tags": ["Load Testing", "QA", "Monitoring"], "priority": "low", "story_points": 5},
        {"title": "Create test data factories", "description": "Build reusable test fixtures and factory functions for all models", "tags": ["Jest", "Unit Testing", "Integration Testing"], "priority": "medium", "story_points": 3},
    ],
    "design": [
        {"title": "Design UI kit", "description": "Create comprehensive UI kit with reusable component designs", "tags": ["Figma", "UI Design", "Design Systems"], "priority": "medium", "story_points": 5},
        {"title": "Create AI feature wireframes", "description": "Design wireframes for AI-powered decomposition, assignment, and sprint risk panels", "tags": ["Wireframing", "UX Research", "Figma", "UI Design"], "priority": "high", "story_points": 5},
        {"title": "Design mobile layout", "description": "Adapt the PM interface for mobile with touch-friendly interactions", "tags": ["UI Design", "Figma", "Prototyping", "Accessibility"], "priority": "low", "story_points": 5},
        {"title": "Create icon set", "description": "Design custom icons for status and priority indicators", "tags": ["Figma", "UI Design", "Design Systems"], "priority": "low", "story_points": 3},
        {"title": "User research for AI features", "description": "Conduct user interviews to understand expectations for AI suggestions", "tags": ["UX Research", "Prototyping", "Wireframing"], "priority": "medium", "story_points": 3},
    ],
    "documentation": [
        {"title": "Write API documentation", "description": "Create comprehensive API docs with request/response examples and error codes", "tags": ["API Docs", "Swagger", "Technical Writing", "Documentation"], "priority": "medium", "story_points": 5},
        {"title": "Create setup guide", "description": "Write getting-started guide for new developers covering all services", "tags": ["Technical Writing", "Markdown", "Documentation"], "priority": "high", "story_points": 3},
        {"title": "Write architecture docs", "description": "Document key architectural decisions and technology choices", "tags": ["Technical Writing", "Documentation", "Confluence"], "priority": "low", "story_points": 3},
        {"title": "Create user guide", "description": "Write end-user documentation for the project management features", "tags": ["Technical Writing", "Documentation", "Tutorials"], "priority": "low", "story_points": 5},
    ],
}

# --- Member profile templates ---
MEMBER_TEMPLATES = [
    {"name": "Frontend Lead",    "role": "lead",        "skills": ["React", "TypeScript", "CSS", "Tailwind", "Next.js", "JavaScript", "HTML"], "skill_levels": {"React": "expert", "TypeScript": "expert", "CSS": "expert"}},
    {"name": "Backend Dev",      "role": "contributor",  "skills": ["Python", "PostgreSQL", "FastAPI", "REST API", "SQL", "Redis"], "skill_levels": {"Python": "expert", "PostgreSQL": "expert"}},
    {"name": "Full Stack Dev 1", "role": "contributor",  "skills": ["React", "Node.js", "TypeScript", "PostgreSQL", "REST API", "JavaScript"], "skill_levels": {"React": "intermediate", "Node.js": "expert"}},
    {"name": "Full Stack Dev 2", "role": "contributor",  "skills": ["React", "Python", "TypeScript", "Django", "CSS", "SQL"], "skill_levels": {"Python": "intermediate", "React": "intermediate"}},
    {"name": "DevOps Engineer",  "role": "contributor",  "skills": ["Docker", "Kubernetes", "CI/CD", "AWS", "Linux", "Terraform", "GitHub Actions", "Nginx", "Monitoring", "Shell"], "skill_levels": {"Docker": "expert", "AWS": "expert"}},
    {"name": "QA Engineer",      "role": "contributor",  "skills": ["Jest", "Playwright", "Cypress", "Unit Testing", "Integration Testing", "E2E", "QA", "Selenium", "Load Testing"], "skill_levels": {"Playwright": "expert", "Jest": "expert"}},
    {"name": "UI Designer",      "role": "contributor",  "skills": ["Figma", "UI Design", "UX Research", "Wireframing", "Prototyping", "Design Systems", "Accessibility", "CSS"], "skill_levels": {"Figma": "expert", "UI Design": "expert"}},
    {"name": "Tech Writer",      "role": "contributor",  "skills": ["Technical Writing", "API Docs", "Markdown", "Confluence", "Swagger", "Documentation", "Tutorials"], "skill_levels": {"Technical Writing": "expert"}},
    {"name": "Junior Dev 1",     "role": "contributor",  "skills": ["JavaScript", "HTML", "CSS", "React"], "skill_levels": {"JavaScript": "beginner", "React": "beginner"}},
    {"name": "Junior Dev 2",     "role": "contributor",  "skills": ["Python", "SQL", "Node.js"], "skill_levels": {"Python": "beginner", "Node.js": "beginner"}},
    {"name": "Team Lead",        "role": "lead",         "skills": ["React", "Python", "Node.js", "PostgreSQL", "Docker", "CI/CD", "TypeScript", "REST API", "AWS"], "skill_levels": {"React": "expert", "Python": "expert", "Node.js": "expert"}},
    {"name": "Project Owner",    "role": "owner",        "skills": ["Technical Writing", "UX Research", "REST API"], "skill_levels": {"Technical Writing": "intermediate"}},
]

# --- Role → category suitability mapping ---
ROLE_CATEGORY_FIT = {
    # (member_template_name, task_category) → role_match_score
    "Frontend Lead": {"frontend": 1.0, "backend": 0.3, "testing": 0.5, "design": 0.6, "devops": 0.1, "documentation": 0.3},
    "Backend Dev": {"frontend": 0.1, "backend": 1.0, "testing": 0.5, "design": 0.0, "devops": 0.4, "documentation": 0.3},
    "Full Stack Dev 1": {"frontend": 0.8, "backend": 0.8, "testing": 0.5, "design": 0.2, "devops": 0.2, "documentation": 0.3},
    "Full Stack Dev 2": {"frontend": 0.7, "backend": 0.7, "testing": 0.4, "design": 0.3, "devops": 0.2, "documentation": 0.3},
    "DevOps Engineer": {"frontend": 0.0, "backend": 0.3, "testing": 0.3, "design": 0.0, "devops": 1.0, "documentation": 0.3},
    "QA Engineer": {"frontend": 0.2, "backend": 0.2, "testing": 1.0, "design": 0.1, "devops": 0.2, "documentation": 0.3},
    "UI Designer": {"frontend": 0.5, "backend": 0.0, "testing": 0.1, "design": 1.0, "devops": 0.0, "documentation": 0.3},
    "Tech Writer": {"frontend": 0.1, "backend": 0.1, "testing": 0.1, "design": 0.2, "devops": 0.1, "documentation": 1.0},
    "Junior Dev 1": {"frontend": 0.5, "backend": 0.2, "testing": 0.3, "design": 0.2, "devops": 0.1, "documentation": 0.2},
    "Junior Dev 2": {"frontend": 0.2, "backend": 0.5, "testing": 0.3, "design": 0.0, "devops": 0.1, "documentation": 0.2},
    "Team Lead": {"frontend": 0.7, "backend": 0.7, "testing": 0.4, "design": 0.3, "devops": 0.5, "documentation": 0.4},
    "Project Owner": {"frontend": 0.2, "backend": 0.2, "testing": 0.2, "design": 0.4, "devops": 0.2, "documentation": 0.6},
}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_ground_truth_score(
    task: dict,
    member: dict,
    category: str,
    all_members: list[dict],
) -> float:
    """Compute ground-truth assignment score using the weighted rule system.

    Returns a score in [0, 1] — the highest-scoring member should be assigned.
    """
    # 1. Skill match (weight: 0.40)
    task_tags = set(tag.lower() for tag in task["tags"])
    member_skills = set(s.lower() for s in member["skills"])
    skill_match = jaccard_similarity(task_tags, member_skills)

    # 2. Workload (weight: 0.25) — lower is better
    max_workload = max(m["current_workload"] for m in all_members) or 1
    workload_score = 1.0 - (member["current_workload"] / (max_workload + 1))

    # 3. Role appropriateness (weight: 0.20)
    role_fit = ROLE_CATEGORY_FIT.get(member["template_name"], {}).get(category, 0.3)

    # 4. Past performance (weight: 0.15) — higher is better
    max_completed = max(m["completed_last_30d"] for m in all_members) or 1
    performance = member["completed_last_30d"] / max_completed

    # Weighted sum
    score = (
        0.40 * skill_match
        + 0.25 * workload_score
        + 0.20 * role_fit
        + 0.15 * performance
    )
    return score


def generate_examples(n_examples: int = 5000) -> list[dict]:
    """Generate n training examples as (task, members, correct_assignee) triples."""
    examples = []

    for _ in range(n_examples):
        # Pick a random category and task
        category = random.choice(list(TASK_TEMPLATES.keys()))
        task = random.choice(TASK_TEMPLATES[category]).copy()

        # Slight variations
        task["story_points"] = random.choice([1, 2, 3, 5, 8])
        task["priority"] = random.choice(["low", "medium", "high", "urgent"])

        # Randomly add/remove tags for variety
        if random.random() > 0.7:
            extra_skill = random.choice(SKILL_POOLS.get(category, ALL_SKILLS))
            task["tags"] = list(set(task["tags"] + [extra_skill]))
        if random.random() > 0.8 and len(task["tags"]) > 2:
            task["tags"] = task["tags"][:-1]

        # Select 3-8 random members for this team
        team_size = random.randint(3, 8)
        team_templates = random.sample(MEMBER_TEMPLATES, min(team_size, len(MEMBER_TEMPLATES)))

        members = []
        for tpl in team_templates:
            member = {
                "template_name": tpl["name"],
                "user_id": f"user-{tpl['name'].lower().replace(' ', '-')}",
                "role": tpl["role"],
                "skills": tpl["skills"].copy(),
                "current_workload": random.randint(0, 40),      # story points
                "completed_last_30d": random.randint(2, 25),
            }

            # Occasionally add or remove a skill for variety
            if random.random() > 0.7 and len(member["skills"]) > 2:
                member["skills"].pop(random.randint(0, len(member["skills"]) - 1))
            if random.random() > 0.8:
                member["skills"].append(random.choice(ALL_SKILLS))
                member["skills"] = list(set(member["skills"]))

            members.append(member)

        # Compute ground truth: member with highest score wins
        scores = []
        for m in members:
            s = compute_ground_truth_score(task, m, category, members)
            scores.append(s)

        best_idx = int(np.argmax(scores))

        # Add some noise: occasionally (10%) pick the 2nd best
        if random.random() < 0.10 and len(scores) > 1:
            sorted_indices = np.argsort(scores)
            best_idx = sorted_indices[-2]

        examples.append({
            "task": task,
            "category": category,
            "members": members,
            "correct_assignee_idx": best_idx,
            "scores": scores,
        })

    return examples


print("Generating 5000 training examples...")
all_examples = generate_examples(5000)

# Stats
cats = pd.Series([e["category"] for e in all_examples]).value_counts()
team_sizes = [len(e["members"]) for e in all_examples]
print(f"Generated {len(all_examples)} examples")
print(f"\nCategory distribution:\n{cats.to_string()}")
print(f"\nTeam sizes: min={min(team_sizes)}, max={max(team_sizes)}, mean={np.mean(team_sizes):.1f}")


# ============================================================================
# SECTION 3: Feature Engineering
# ============================================================================
# MARKDOWN: ## 3. Feature Engineering
#
# For each (task, member) pair we compute a 6-dimensional feature vector:
#
# | Feature | Computation |
# |---------|-------------|
# | `skill_match_score` | Jaccard similarity between task tags and member skills |
# | `semantic_similarity` | Sentence-BERT cosine sim: task description vs skill list string |
# | `workload_score` | `1 - (current_story_points / max_capacity)`, normalized |
# | `role_match_score` | Category-role fit lookup, 0.0 to 1.0 |
# | `performance_score` | `completed_tasks_last_30d / max_in_team`, normalized |
# | `availability_score` | Based on workload thresholds: <15 SP=1.0, <30=0.5, else 0.0 |
#
# Uses `all-MiniLM-L6-v2` (22MB, CPU-friendly) for semantic similarity.

print("\n" + "="*60)
print("SECTION 3: Feature engineering")
print("="*60)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# Load sentence-BERT model
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading Sentence-BERT model: {SBERT_MODEL_NAME}...")
sbert = SentenceTransformer(SBERT_MODEL_NAME)
print(f"Loaded. Embedding dimension: {sbert.get_sentence_embedding_dimension()}")

# Workload capacity constant
MAX_CAPACITY = 40  # story points — typical 2-week sprint max


def build_task_text(task: dict) -> str:
    """Build a text string from task for embedding."""
    parts = [task["title"]]
    if task.get("description"):
        parts.append(task["description"])
    if task.get("tags"):
        parts.append("Tags: " + ", ".join(task["tags"]))
    return " ".join(parts)


def build_member_text(member: dict) -> str:
    """Build a text string from member skills for embedding."""
    return f"Skills: {', '.join(member['skills'])}. Role: {member['role']}"


def compute_features(task: dict, member: dict, category: str,
                     all_members: list[dict],
                     task_embedding: np.ndarray,
                     member_embedding: np.ndarray) -> np.ndarray:
    """Compute the 6-dimensional feature vector for a (task, member) pair."""

    # 1. Skill match — Jaccard similarity
    task_tags = set(t.lower() for t in task["tags"])
    member_skills = set(s.lower() for s in member["skills"])
    skill_match = jaccard_similarity(task_tags, member_skills)

    # 2. Semantic similarity — SBERT cosine
    semantic_sim = float(cos_sim(
        task_embedding.reshape(1, -1),
        member_embedding.reshape(1, -1),
    )[0, 0])

    # 3. Workload score — normalized inverse
    workload_score = max(0.0, 1.0 - (member["current_workload"] / MAX_CAPACITY))

    # 4. Role match score — category-role lookup
    role_match = ROLE_CATEGORY_FIT.get(
        member.get("template_name", ""), {}
    ).get(category, 0.3)

    # 5. Performance score — normalized
    max_completed = max((m["completed_last_30d"] for m in all_members), default=1) or 1
    performance = member["completed_last_30d"] / max_completed

    # 6. Availability score — threshold-based
    wl = member["current_workload"]
    if wl < 15:
        availability = 1.0
    elif wl < 30:
        availability = 0.5
    else:
        availability = 0.0

    return np.array([
        skill_match,
        semantic_sim,
        workload_score,
        role_match,
        performance,
        availability,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "skill_match_score",
    "semantic_similarity",
    "workload_score",
    "role_match_score",
    "performance_score",
    "availability_score",
]

# --- Build feature matrix ---
print("\nComputing Sentence-BERT embeddings and features...")
start_time = time.time()

# Pre-compute all unique text embeddings (batch for efficiency)
unique_task_texts = list({build_task_text(e["task"]) for e in all_examples})
unique_member_texts = list({build_member_text(m) for e in all_examples for m in e["members"]})

print(f"  Unique task texts: {len(unique_task_texts)}")
print(f"  Unique member texts: {len(unique_member_texts)}")

task_embeddings_map = {}
print("  Encoding task texts...")
task_embs = sbert.encode(unique_task_texts, batch_size=128, show_progress_bar=False)
for text, emb in zip(unique_task_texts, task_embs):
    task_embeddings_map[text] = emb

member_embeddings_map = {}
print("  Encoding member texts...")
member_embs = sbert.encode(unique_member_texts, batch_size=128, show_progress_bar=False)
for text, emb in zip(unique_member_texts, member_embs):
    member_embeddings_map[text] = emb

embed_time = time.time() - start_time
print(f"  Embeddings computed in {embed_time:.1f}s")

# Build training data: one row per (task, member) pair
# Label: 1 if this member is the correct assignee, 0 otherwise
X_rows = []
y_rows = []

print("  Building feature matrix...")
for example in all_examples:
    task = example["task"]
    category = example["category"]
    members = example["members"]
    correct_idx = example["correct_assignee_idx"]

    task_text = build_task_text(task)
    task_emb = task_embeddings_map[task_text]

    for member_idx, member in enumerate(members):
        member_text = build_member_text(member)
        member_emb = member_embeddings_map[member_text]

        features = compute_features(
            task, member, category, members, task_emb, member_emb
        )
        X_rows.append(features)
        y_rows.append(1 if member_idx == correct_idx else 0)

X = np.array(X_rows, dtype=np.float32)
y = np.array(y_rows, dtype=np.int32)

total_time = time.time() - start_time
print(f"\nFeature matrix shape: {X.shape}")
print(f"Positive samples (correct assignee): {y.sum()} ({y.mean()*100:.1f}%)")
print(f"Negative samples: {(1-y).sum()} ({(1-y).mean()*100:.1f}%)")
print(f"Total feature engineering time: {total_time:.1f}s")

# Feature statistics
feature_df = pd.DataFrame(X, columns=FEATURE_NAMES)
print(f"\nFeature statistics:")
print(feature_df.describe().round(3).to_string())


# ============================================================================
# SECTION 4: Model Training
# ============================================================================
# MARKDOWN: ## 4. Model Training
#
# We train two models and pick the better one:
# - **Model A:** GradientBoostingClassifier (n_estimators=100, max_depth=4)
# - **Model B:** RandomForestClassifier (n_estimators=200)
#
# Evaluation via 5-fold cross-validation.
# Target: >85% accuracy on synthetic data.

print("\n" + "="*60)
print("SECTION 4: Model training")
print("="*60)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED, stratify=y,
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# --- Model A: Gradient Boosting ---
print("\n--- Model A: GradientBoostingClassifier ---")
model_a = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=SEED,
)

print("  5-fold cross-validation...")
cv_scores_a = cross_val_score(model_a, X_train, y_train, cv=5, scoring="f1")
print(f"  CV F1 scores: {cv_scores_a.round(4)}")
print(f"  CV F1 mean:   {cv_scores_a.mean():.4f} (+/- {cv_scores_a.std():.4f})")

print("  Training on full train set...")
t0 = time.time()
model_a.fit(X_train, y_train)
train_time_a = time.time() - t0
print(f"  Training time: {train_time_a:.2f}s")

y_pred_a = model_a.predict(X_test)
acc_a = accuracy_score(y_test, y_pred_a)
prec_a = precision_score(y_test, y_pred_a)
rec_a = recall_score(y_test, y_pred_a)
f1_a = f1_score(y_test, y_pred_a)
print(f"  Test accuracy:  {acc_a:.4f}")
print(f"  Test precision: {prec_a:.4f}")
print(f"  Test recall:    {rec_a:.4f}")
print(f"  Test F1:        {f1_a:.4f}")

# --- Model B: Random Forest ---
print("\n--- Model B: RandomForestClassifier ---")
model_b = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=SEED,
    n_jobs=-1,
)

print("  5-fold cross-validation...")
cv_scores_b = cross_val_score(model_b, X_train, y_train, cv=5, scoring="f1")
print(f"  CV F1 scores: {cv_scores_b.round(4)}")
print(f"  CV F1 mean:   {cv_scores_b.mean():.4f} (+/- {cv_scores_b.std():.4f})")

print("  Training on full train set...")
t0 = time.time()
model_b.fit(X_train, y_train)
train_time_b = time.time() - t0
print(f"  Training time: {train_time_b:.2f}s")

y_pred_b = model_b.predict(X_test)
acc_b = accuracy_score(y_test, y_pred_b)
prec_b = precision_score(y_test, y_pred_b)
rec_b = recall_score(y_test, y_pred_b)
f1_b = f1_score(y_test, y_pred_b)
print(f"  Test accuracy:  {acc_b:.4f}")
print(f"  Test precision: {prec_b:.4f}")
print(f"  Test recall:    {rec_b:.4f}")
print(f"  Test F1:        {f1_b:.4f}")

# --- Select the better model ---
print("\n--- Model Comparison ---")
print(f"  GradientBoosting F1: {f1_a:.4f}  |  RandomForest F1: {f1_b:.4f}")

if f1_a >= f1_b:
    best_model = model_a
    best_name = "GradientBoosting"
    best_f1 = f1_a
    best_acc = acc_a
    best_prec = prec_a
    best_rec = rec_a
    y_pred_best = y_pred_a
    print(f"  Winner: GradientBoostingClassifier")
else:
    best_model = model_b
    best_name = "RandomForest"
    best_f1 = f1_b
    best_acc = acc_b
    best_prec = prec_b
    best_rec = rec_b
    y_pred_best = y_pred_b
    print(f"  Winner: RandomForestClassifier")

# Detailed classification report
print(f"\n--- Classification Report ({best_name}) ---")
print(classification_report(y_test, y_pred_best, target_names=["Not Assignee", "Correct Assignee"]))

# Feature importances
print(f"\n--- Feature Importances ({best_name}) ---")
importances = best_model.feature_importances_
for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
    bar = "#" * int(imp * 50)
    print(f"  {name:25s} {imp:.4f} {bar}")


# ============================================================================
# SECTION 5: Calibration
# ============================================================================
# MARKDOWN: ## 5. Probability Calibration
#
# Scikit-learn classifiers don't always produce well-calibrated probabilities.
# We use `CalibratedClassifierCV` with sigmoid method to ensure the
# `predict_proba()` values are meaningful confidence percentages.

print("\n" + "="*60)
print("SECTION 5: Probability calibration")
print("="*60)

print("Calibrating with CalibratedClassifierCV (sigmoid, 3-fold)...")
calibrated_model = CalibratedClassifierCV(
    best_model,
    method="sigmoid",
    cv=3,
)
calibrated_model.fit(X_train, y_train)

# Test calibrated probabilities
y_proba = calibrated_model.predict_proba(X_test)[:, 1]
y_pred_cal = (y_proba >= 0.5).astype(int)

cal_acc = accuracy_score(y_test, y_pred_cal)
cal_f1 = f1_score(y_test, y_pred_cal)
print(f"\nCalibrated model — Test accuracy: {cal_acc:.4f}, F1: {cal_f1:.4f}")

# Probability distribution
print(f"\nProbability distribution for positive samples:")
pos_probs = y_proba[y_test == 1]
neg_probs = y_proba[y_test == 0]
print(f"  Positive (correct assignee): mean={pos_probs.mean():.3f}, std={pos_probs.std():.3f}, min={pos_probs.min():.3f}, max={pos_probs.max():.3f}")
print(f"  Negative (not assignee):     mean={neg_probs.mean():.3f}, std={neg_probs.std():.3f}, min={neg_probs.min():.3f}, max={neg_probs.max():.3f}")

# Verify calibration quality: probability should track actual rate
print(f"\nCalibration check (grouped by predicted probability):")
for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
    mask = (y_proba >= lo) & (y_proba < hi)
    if mask.sum() > 0:
        actual_rate = y_test[mask].mean()
        print(f"  P({lo:.1f}-{hi:.1f}): {mask.sum():5d} samples, actual positive rate: {actual_rate:.3f}")

# End-to-end test: simulate assigning a task to a team
print("\n--- End-to-end Assignment Test ---")
test_example = all_examples[0]
test_task = test_example["task"]
test_members = test_example["members"]
test_category = test_example["category"]

task_text = build_task_text(test_task)
task_emb = sbert.encode([task_text])[0]

member_scores = []
for member in test_members:
    member_text = build_member_text(member)
    member_emb = sbert.encode([member_text])[0]
    features = compute_features(
        test_task, member, test_category, test_members, task_emb, member_emb
    )
    features_scaled = scaler.transform(features.reshape(1, -1))
    prob = calibrated_model.predict_proba(features_scaled)[0, 1]
    member_scores.append((member["template_name"], prob, member["skills"][:3]))

member_scores.sort(key=lambda x: -x[1])
print(f"Task: {test_task['title']} ({test_category})")
print(f"Tags: {test_task['tags']}")
print(f"\nRanked suggestions:")
for rank, (name, prob, skills) in enumerate(member_scores, 1):
    bar = "#" * int(prob * 30)
    print(f"  {rank}. {name:20s} confidence: {prob:.2%}  skills: {skills}  {bar}")


# ============================================================================
# SECTION 6: Save & Export
# ============================================================================
# MARKDOWN: ## 6. Save & Export
#
# We save:
# - Calibrated model: `assigner_model.pkl`
# - Feature scaler: `assigner_scaler.pkl`
# - Metadata JSON with model name, feature names, and performance metrics
#
# The Sentence-BERT model is referenced by name (downloaded at inference time
# from HuggingFace cache) — we don't save its weights again.

print("\n" + "="*60)
print("SECTION 6: Save & Export")
print("="*60)

SAVE_DIR = Path("/content/assigner-model") if Path("/content").exists() else Path("./assigner-model")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Save calibrated model
model_path = SAVE_DIR / "assigner_model.pkl"
joblib.dump(calibrated_model, model_path)
model_size = model_path.stat().st_size
print(f"Saved calibrated model: {model_path} ({model_size / 1e6:.2f} MB)")

# Save scaler
scaler_path = SAVE_DIR / "assigner_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"Saved scaler: {scaler_path} ({scaler_path.stat().st_size / 1e3:.1f} KB)")

# Benchmark inference time
print("\nBenchmarking inference time (team of 10)...")
dummy_features = np.random.randn(10, 6).astype(np.float32)
dummy_scaled = scaler.transform(dummy_features)

# Warm up
for _ in range(10):
    calibrated_model.predict_proba(dummy_scaled)

# Time 100 runs
times = []
for _ in range(100):
    t0 = time.perf_counter()
    calibrated_model.predict_proba(dummy_scaled)
    times.append(time.perf_counter() - t0)

avg_ms = np.mean(times) * 1000
p99_ms = np.percentile(times, 99) * 1000
print(f"  Inference for 10 members: avg={avg_ms:.2f}ms, p99={p99_ms:.2f}ms")

# Include SBERT encoding time estimate
t0 = time.perf_counter()
for _ in range(10):
    sbert.encode(["Test task about building a React dashboard with charts"])
sbert_ms = (time.perf_counter() - t0) / 10 * 1000
print(f"  SBERT encode (1 text): ~{sbert_ms:.1f}ms")
print(f"  Total estimated per request (10 members): ~{avg_ms + sbert_ms * 11:.0f}ms")

target_met = (avg_ms + sbert_ms * 11) < 50
if target_met:
    print(f"  TARGET MET: <50ms per prediction")
else:
    print(f"  Note: Total inference >50ms. Consider caching SBERT embeddings for members.")

# Save metadata
metadata = {
    "model_type": best_name,
    "model_version": "assigner-v1",
    "calibrated": True,
    "sbert_model": SBERT_MODEL_NAME,
    "feature_names": FEATURE_NAMES,
    "feature_count": len(FEATURE_NAMES),
    "training_examples": len(all_examples),
    "total_feature_rows": len(X),
    "test_accuracy": round(cal_acc, 4),
    "test_f1": round(cal_f1, 4),
    "test_precision": round(float(precision_score(y_test, y_pred_cal)), 4),
    "test_recall": round(float(recall_score(y_test, y_pred_cal)), 4),
    "feature_importances": {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_NAMES, importances)
    },
    "scoring_weights": {
        "skill_match": 0.40,
        "workload": 0.25,
        "role_match": 0.20,
        "performance": 0.15,
    },
    "inference_time_ms": round(avg_ms, 2),
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
    label = f"{size/1e6:.2f} MB" if size > 1e6 else f"{size/1e3:.1f} KB"
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

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"\nModel: {best_name} (calibrated)")
print(f"Accuracy: {cal_acc:.4f}")
print(f"F1 Score: {cal_f1:.4f}")
print(f"Inference: {avg_ms:.2f}ms for 10 members")
print(f"\nTo deploy in AI server:")
print(f"  1. Unzip assigner-model.zip")
print(f"  2. Copy contents to ai-server/model_weights/assigner/")
print(f"  3. Restart the AI server")
print(f"  4. /health should show \"assigner\": true")
