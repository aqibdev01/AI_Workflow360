"""Inference logic for task decomposition using FLAN-T5.

Real mode: builds a structured prompt, runs FLAN-T5 generation, parses
the output lines in SUBTASK_N format.

Mock mode: returns realistic fake subtask suggestions based on keyword
matching so the full UI can be built before training is complete.
"""

import logging
import random
import re

from .model import get_model
from utils.privacy import assert_no_pii

log = logging.getLogger(__name__)

_FIBONACCI = {1, 2, 3, 5, 8, 13}


def decompose(
    task_id: str,
    title: str,
    description: str,
    priority: str,
    project_context: str,
    existing_tags: list[str],
) -> dict:
    """Run decomposition inference (or mock).

    Returns a dict matching DecomposeResponse schema.
    """
    # PII hard-stop — model never runs if this fails
    assert_no_pii(
        {
            "task_id": task_id,
            "title": title,
            "description": description,
            "priority": priority,
            "project_context": project_context,
            "existing_tags": existing_tags,
        },
        context="decomposition",
    )

    dm = get_model()

    if dm.is_mock:
        return _mock_decompose(task_id, title, description, priority, existing_tags)

    return _real_decompose(
        dm, task_id, title, description, priority, existing_tags, project_context
    )


# ---------------------------------------------------------------------------
# Real model inference
# ---------------------------------------------------------------------------
def _real_decompose(
    dm, task_id, title, description, priority, tags, project_context
) -> dict:
    """Run FLAN-T5 generation and parse structured output."""
    import torch

    device = next(dm.model.parameters()).device

    # Build prompt matching training format
    prompt = (
        f"Decompose this software task into subtasks:\n"
        f"Task: {title}\n"
        f"Description: {description}\n"
        f"Priority: {priority}\n"
        f"Context: {project_context or 'software project'}"
    )

    inputs = dm.tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = dm.model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            temperature=0.7,
            early_stopping=True,
            do_sample=False,
        )

    raw_output = dm.tokenizer.decode(outputs[0], skip_special_tokens=True)
    log.debug("Raw model output: %s", raw_output)

    # Parse structured lines
    subtasks = _parse_model_output(raw_output, priority, tags)
    overall_confidence = _compute_confidence(subtasks, raw_output)

    return {
        "task_id": task_id,
        "subtasks": subtasks,
        "overall_confidence": overall_confidence,
        "model_version": dm.version,
    }


# ---------------------------------------------------------------------------
# Output parsing — handles SUBTASK_N structured format
# ---------------------------------------------------------------------------
_SUBTASK_PATTERN = re.compile(
    r"SUBTASK_\d+:\s*(?P<title>[^|]+)"
    r"\|\s*PRIORITY:\s*(?P<priority>[^|]+)"
    r"\|\s*POINTS:\s*(?P<points>[^|]+)"
    r"\|\s*DAYS:\s*(?P<days>[^|]+)"
    r"\|\s*TAGS:\s*(?P<tags>.+)",
    re.IGNORECASE,
)


def parse_subtask_line(line: str) -> dict | None:
    """Parse a single SUBTASK_N line into a subtask dict.

    Expected format:
      SUBTASK_1: title | PRIORITY: p | POINTS: n | DAYS: n | TAGS: t1,t2

    Returns None if the line doesn't match the expected format.
    """
    match = _SUBTASK_PATTERN.match(line.strip())
    if not match:
        return None

    title = match.group("title").strip()
    priority = match.group("priority").strip().lower()
    tags_str = match.group("tags").strip()

    # Parse story points — must be Fibonacci
    try:
        points = int(float(match.group("points").strip()))
        if points not in _FIBONACCI:
            points = min(_FIBONACCI, key=lambda x: abs(x - points))
    except (ValueError, TypeError):
        points = 2

    # Parse days
    try:
        days = round(float(match.group("days").strip()), 1)
        days = max(0.5, min(20.0, days))
    except (ValueError, TypeError):
        days = 1.0

    # Parse tags
    tags = [t.strip() for t in tags_str.split(",") if t.strip()]

    # Validate priority
    valid_priorities = {"low", "medium", "high", "urgent"}
    if priority not in valid_priorities:
        priority = "medium"

    return {
        "title": title,
        "description": "",
        "priority": priority,
        "story_points": points,
        "estimated_days": days,
        "tags": tags,
        "confidence": round(random.uniform(0.65, 0.92), 2),
    }


def _parse_model_output(
    raw: str, fallback_priority: str, fallback_tags: list[str]
) -> list[dict]:
    """Parse full model output into a list of subtask dicts.

    Tries structured SUBTASK_N format first, then falls back to
    line-by-line parsing for unstructured output.
    """
    subtasks = []

    # Try structured format first
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        parsed = parse_subtask_line(line)
        if parsed:
            subtasks.append(parsed)

    if subtasks:
        return subtasks

    # Fallback: treat each non-empty line as a subtask title
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    for line in lines:
        # Strip leading numbering like "1." or "1)"
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        if cleaned and len(cleaned) > 3:
            subtasks.append({
                "title": cleaned[:120],
                "description": "",
                "priority": fallback_priority,
                "story_points": 2,
                "estimated_days": 1.0,
                "tags": fallback_tags[:2] if fallback_tags else ["general"],
                "confidence": 0.45,
            })

    # If nothing parsed at all, return a single generic subtask
    if not subtasks:
        subtasks.append({
            "title": "Implement task requirements",
            "description": "",
            "priority": fallback_priority,
            "story_points": 3,
            "estimated_days": 1.5,
            "tags": fallback_tags[:2] if fallback_tags else ["general"],
            "confidence": 0.3,
        })

    return subtasks


def _compute_confidence(subtasks: list[dict], raw_output: str) -> float:
    """Compute overall confidence based on output quality signals."""
    if not subtasks:
        return 0.1

    score = 0.4

    # More subtasks (2-6) is a good sign
    n = len(subtasks)
    if 2 <= n <= 6:
        score += 0.15
    elif n > 6:
        score += 0.05

    # Structured format parsed successfully
    if any(parse_subtask_line(line) for line in raw_output.split("\n") if line.strip()):
        score += 0.2

    # Subtasks have varied story points
    points = {s.get("story_points") for s in subtasks}
    if len(points) > 1:
        score += 0.1

    # Individual confidence average
    avg_conf = sum(s.get("confidence", 0.5) for s in subtasks) / n
    score += avg_conf * 0.15

    return max(0.0, min(1.0, round(score, 2)))


# ---------------------------------------------------------------------------
# Mock inference — realistic fake data for UI development
# ---------------------------------------------------------------------------
_MOCK_KEYWORD_TEMPLATES = {
    "auth": [
        ("Set up authentication provider configuration", "high", 3, 1.0, "backend,auth"),
        ("Implement login and signup API endpoints", "high", 5, 2.0, "backend,auth,api"),
        ("Build login form with validation", "medium", 3, 1.0, "frontend,auth,forms"),
        ("Add session management and token refresh", "high", 3, 1.5, "backend,auth,security"),
        ("Write authentication tests", "medium", 3, 1.0, "testing,auth"),
    ],
    "api": [
        ("Design API endpoint structure and request schemas", "high", 2, 0.5, "backend,api,design"),
        ("Implement CRUD endpoints with validation", "high", 5, 2.0, "backend,api"),
        ("Add error handling and status codes", "medium", 2, 1.0, "backend,api"),
        ("Write API integration tests", "medium", 3, 1.5, "testing,api"),
        ("Document API endpoints with examples", "low", 2, 1.0, "docs,api"),
    ],
    "ui": [
        ("Create component layout and structure", "medium", 2, 0.5, "frontend,ui"),
        ("Build interactive UI components", "high", 5, 2.0, "frontend,ui"),
        ("Add responsive design and mobile support", "medium", 3, 1.0, "frontend,ui,responsive"),
        ("Implement accessibility features", "low", 2, 1.0, "frontend,ui,a11y"),
    ],
    "database": [
        ("Design database schema and relations", "high", 3, 1.0, "backend,database"),
        ("Write migration scripts", "high", 2, 0.5, "backend,database,migration"),
        ("Add indexes for query optimization", "medium", 2, 0.5, "backend,database"),
        ("Create seed data for development", "low", 2, 1.0, "backend,database"),
    ],
    "test": [
        ("Set up test framework and utilities", "high", 2, 0.5, "testing,setup"),
        ("Write unit tests for core logic", "high", 3, 1.5, "testing,unit"),
        ("Write integration tests", "medium", 5, 2.0, "testing,integration"),
        ("Set up CI test pipeline", "medium", 2, 1.0, "testing,ci"),
    ],
    "deploy": [
        ("Write Dockerfile and container config", "high", 3, 1.0, "devops,docker"),
        ("Configure CI/CD pipeline", "high", 3, 1.5, "devops,ci"),
        ("Set up environment variables and secrets", "medium", 2, 0.5, "devops,config"),
        ("Add health checks and monitoring", "medium", 2, 1.0, "devops,monitoring"),
    ],
}

# Default template when no keyword matches
_MOCK_DEFAULT_TEMPLATES = {
    "high": [
        ("Define requirements and acceptance criteria", "high", 2, 0.5, "planning"),
        ("Implement core functionality", "high", 5, 2.0, "development"),
        ("Write automated tests", "medium", 3, 1.0, "testing"),
        ("Code review and refactor", "medium", 2, 0.5, "review"),
        ("Deploy and verify", "low", 1, 0.5, "devops"),
    ],
    "medium": [
        ("Research and design approach", "medium", 2, 0.5, "planning"),
        ("Implement solution", "medium", 3, 1.5, "development"),
        ("Test and validate", "medium", 2, 1.0, "testing"),
    ],
    "low": [
        ("Investigate and plan", "low", 1, 0.5, "planning"),
        ("Implement change", "low", 2, 1.0, "development"),
        ("Verify and document", "low", 1, 0.5, "docs"),
    ],
    "urgent": [
        ("Triage and identify root cause", "urgent", 2, 0.5, "investigation"),
        ("Implement fix", "urgent", 3, 1.0, "development"),
        ("Write regression test", "high", 2, 0.5, "testing"),
        ("Deploy hotfix", "urgent", 1, 0.5, "devops"),
    ],
}


def _mock_decompose(
    task_id: str,
    title: str,
    description: str,
    priority: str,
    tags: list[str],
) -> dict:
    """Generate realistic mock subtasks based on keyword matching."""
    log.info("MOCK decomposition for task %s: %s", task_id, title)

    combined_text = f"{title} {description}".lower()

    # Try keyword matching first
    templates = None
    for keyword, keyword_templates in _MOCK_KEYWORD_TEMPLATES.items():
        if keyword in combined_text:
            templates = keyword_templates
            break

    # Fall back to priority-based templates
    if templates is None:
        templates = _MOCK_DEFAULT_TEMPLATES.get(
            priority, _MOCK_DEFAULT_TEMPLATES["medium"]
        )

    subtasks = []
    for title_t, prio, points, days, tags_str in templates:
        subtask_tags = tags_str.split(",")
        # Merge with existing tags if any
        if tags:
            subtask_tags = list(set(subtask_tags + tags[:1]))

        subtasks.append({
            "title": title_t,
            "description": "",
            "priority": prio,
            "story_points": points,
            "estimated_days": days,
            "tags": subtask_tags,
            "confidence": round(random.uniform(0.72, 0.93), 2),
        })

    return {
        "task_id": task_id,
        "subtasks": subtasks,
        "overall_confidence": round(random.uniform(0.75, 0.90), 2),
        "model_version": "mock-v0",
    }
