"""Assignment scoring logic using Sentence-BERT + sklearn classifier.

Real mode: computes 6-feature vector per member, runs calibrated classifier,
returns ranked suggestions with confidence scores and scoring breakdowns.

Mock mode: scores by keyword overlap between task tags and member skills.
"""

import logging
import random

import numpy as np

from .model import get_model
from utils.privacy import assert_no_pii

log = logging.getLogger(__name__)

# Workload capacity constants
_MAX_CAPACITY_SP = 40   # max story points per sprint
_MAX_TASK_COUNT = 8     # task count for availability scoring

# Category keywords for role matching
_FRONTEND_KEYWORDS = {"frontend", "ui", "react", "vue", "angular", "css", "tailwind", "html", "next.js", "component", "responsive"}
_BACKEND_KEYWORDS = {"backend", "api", "database", "server", "python", "node.js", "rest", "graphql", "fastapi", "django", "express", "sql", "redis"}
_DEVOPS_KEYWORDS = {"devops", "docker", "deploy", "kubernetes", "ci/cd", "aws", "terraform", "monitoring", "infrastructure", "nginx"}
_TESTING_KEYWORDS = {"testing", "test", "jest", "playwright", "cypress", "qa", "e2e", "unit", "integration", "selenium"}
_DESIGN_KEYWORDS = {"design", "figma", "wireframe", "prototype", "ux", "ui design", "mockup", "accessibility"}
_DOCS_KEYWORDS = {"documentation", "docs", "technical writing", "api docs", "readme", "tutorial"}


def suggest(
    task_id: str,
    title: str,
    description: str,
    priority: str,
    tags: list[str],
    story_points: int | None,
    project_members: list[dict],
) -> dict:
    """Score each team member and return ranked suggestions.

    Returns a dict matching AssignResponse schema.
    """
    assert_no_pii(
        {
            "task_id": task_id,
            "title": title,
            "description": description,
            "priority": priority,
            "tags": tags,
            "project_members": project_members,
        },
        context="assigner",
    )

    am = get_model()

    if am.is_mock:
        return _mock_suggest(task_id, title, tags, project_members)

    return _real_suggest(am, task_id, title, description, tags, project_members)


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_role_match(task_tags: list[str], member_role: str, member_skills: list[str]) -> float:
    """Compute how well a member's role/skills match the task category.

    Returns 0.0 to 1.0.
    """
    tags_lower = {t.lower() for t in task_tags}
    skills_lower = {s.lower() for s in member_skills}
    combined = tags_lower | skills_lower

    # Owners/leads can do anything but are less optimal
    if member_role in ("owner", "lead"):
        return 0.7

    if member_role == "viewer":
        return 0.1

    # Contributor — check category fit
    if tags_lower & _FRONTEND_KEYWORDS and skills_lower & _FRONTEND_KEYWORDS:
        return 1.0
    if tags_lower & _BACKEND_KEYWORDS and skills_lower & _BACKEND_KEYWORDS:
        return 1.0
    if tags_lower & _DEVOPS_KEYWORDS and skills_lower & _DEVOPS_KEYWORDS:
        return 1.0
    if tags_lower & _TESTING_KEYWORDS and skills_lower & _TESTING_KEYWORDS:
        return 1.0
    if tags_lower & _DESIGN_KEYWORDS and skills_lower & _DESIGN_KEYWORDS:
        return 1.0
    if tags_lower & _DOCS_KEYWORDS and skills_lower & _DOCS_KEYWORDS:
        return 1.0

    # Partial match if any overlap
    if tags_lower & skills_lower:
        return 0.6

    return 0.3


# ---------------------------------------------------------------------------
# Real model inference
# ---------------------------------------------------------------------------

def _real_suggest(am, task_id, title, description, tags, members) -> dict:
    """Run SBERT + sklearn classifier for each member, return ranked suggestions."""
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    # Build task text and encode once
    task_text = f"{title} {description} Tags: {', '.join(tags)}"
    task_embedding = am.embedder.encode([task_text])[0]

    # Precompute team maximums for normalization
    max_sp = max((m.get("current_story_points", 0) for m in members), default=1) or 1
    max_completed = max((m.get("completed_tasks_last_30d", 0) for m in members), default=1) or 1

    candidates = []
    for m in members:
        skills = m.get("skills", [])
        member_text = f"Skills: {', '.join(skills)}. Role: {m['role']}"
        member_embedding = am.embedder.encode([member_text])[0]

        # Feature 1: Skill match — Jaccard
        task_tags_lower = {t.lower() for t in tags}
        member_skills_lower = {s.lower() for s in skills}
        skill_match = _jaccard(task_tags_lower, member_skills_lower)

        # Feature 2: Semantic similarity — SBERT cosine
        semantic_sim = float(cos_sim(
            task_embedding.reshape(1, -1),
            member_embedding.reshape(1, -1),
        )[0, 0])

        # Feature 3: Workload score
        current_sp = m.get("current_story_points", 0)
        workload_score = max(0.0, 1.0 - (current_sp / _MAX_CAPACITY_SP))

        # Feature 4: Role match
        role_match = compute_role_match(tags, m["role"], skills)

        # Feature 5: Performance score
        completed = m.get("completed_tasks_last_30d", 0)
        performance = completed / max_completed

        # Feature 6: Availability score
        task_count = m.get("current_task_count", 0)
        if task_count < 5:
            availability = 1.0
        else:
            availability = max(0.0, 1.0 - (task_count - 5) * 0.2)

        # Build feature vector matching training format
        feature_vec = np.array([[
            skill_match,
            semantic_sim,
            workload_score,
            role_match,
            performance,
            availability,
        ]], dtype=np.float32)

        # Scale features
        feature_scaled = am.scaler.transform(feature_vec)

        # Get calibrated probability
        confidence = float(am.scorer.predict_proba(feature_scaled)[0, 1])
        confidence = round(max(0.0, min(1.0, confidence)), 2)

        candidates.append({
            "user_id": m["user_id"],
            "full_name": m.get("full_name", m["user_id"]),
            "confidence": confidence,
            "scoring_breakdown": {
                "skill_match": round(skill_match, 3),
                "semantic_similarity": round(semantic_sim, 3),
                "workload": round(workload_score, 3),
                "role_match": round(role_match, 3),
                "performance": round(performance, 3),
                "availability": round(availability, 3),
            },
        })

    # Sort by confidence descending, return top 3
    candidates.sort(key=lambda c: c["confidence"], reverse=True)

    return {
        "task_id": task_id,
        "suggestions": candidates[:3],
        "model_version": am.version,
    }


# ---------------------------------------------------------------------------
# Mock inference — keyword-based scoring for UI development
# ---------------------------------------------------------------------------

def _mock_suggest(task_id: str, title: str, tags: list[str], members: list[dict]) -> dict:
    """Score members by keyword overlap — realistic enough for UI testing."""
    log.info("MOCK assignment suggestion for task %s", task_id)

    task_keywords = {t.lower() for t in tags}
    task_keywords |= {w.lower() for w in title.split() if len(w) > 3}

    suggestions = []
    for m in members:
        skills = m.get("skills", [])
        skills_lower = {s.lower() for s in skills}

        # Skill overlap as primary signal
        overlap = len(task_keywords & skills_lower)
        total = len(task_keywords | skills_lower) or 1
        skill_match = round(overlap / total, 2)

        # Workload penalty
        current_sp = m.get("current_story_points", 0)
        workload = round(max(0.0, 1.0 - (current_sp / _MAX_CAPACITY_SP)), 2)

        # Role fit
        role_match = compute_role_match(tags, m.get("role", "contributor"), skills)

        # Performance
        max_completed = max((mm.get("completed_tasks_last_30d", 0) for mm in members), default=1) or 1
        performance = round(m.get("completed_tasks_last_30d", 0) / max_completed, 2)

        # Availability
        task_count = m.get("current_task_count", 0)
        availability = round(max(0.0, 1.0 - max(0, task_count - 5) * 0.2), 2)

        # Weighted sum with slight randomness for variety
        confidence = (
            0.40 * skill_match
            + 0.25 * workload
            + 0.20 * role_match
            + 0.15 * performance
        )
        confidence += random.uniform(-0.03, 0.03)
        confidence = round(max(0.05, min(0.98, confidence)), 2)

        suggestions.append({
            "user_id": m["user_id"],
            "full_name": m.get("full_name", f"User {m['user_id'][:8]}"),
            "confidence": confidence,
            "scoring_breakdown": {
                "skill_match": skill_match,
                "semantic_similarity": round(skill_match * 0.9 + random.uniform(0, 0.1), 3),
                "workload": workload,
                "role_match": round(role_match, 3),
                "performance": performance,
                "availability": availability,
            },
        })

    suggestions.sort(key=lambda s: s["confidence"], reverse=True)

    return {
        "task_id": task_id,
        "suggestions": suggestions[:3],
        "model_version": "mock-v0",
    }
