"""POST /api/suggest-assignee — Smart assignment endpoint (M10).

Logs every request to a JSON log file for audit and model evaluation.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import io

from schemas.assigner import AssignRequest, AssignResponse, AssigneeSuggestion
from models.assigner.inference import suggest
from utils.auth import verify_api_key

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# JSON log file for assignment requests
_LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
_LOG_FILE = _LOG_DIR / "assignment_requests.jsonl"


def _log_request(task_id: str, num_suggestions: int, top_confidence: float, model_version: str):
    """Append a JSON log entry for this assignment request."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "num_candidates": num_suggestions,
            "top_confidence": top_confidence,
            "model_version": model_version,
        }
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("Failed to write assignment log: %s", exc)


@router.post(
    "/suggest-assignee",
    summary="Suggest the best assignee for a task",
    description=(
        "Scores all project members using SBERT embeddings + trained classifier. "
        "Returns top 3 suggestions with confidence scores and scoring breakdowns."
    ),
)
async def suggest_assignee(
    req: AssignRequest,
    _key: str = Depends(verify_api_key),
):
    log.info(
        "Assignment request for task %s (%d candidates)",
        req.task_id,
        len(req.project_members),
    )

    members = [m.model_dump() for m in req.project_members]

    result = suggest(
        task_id=req.task_id,
        title=req.title,
        description=req.description,
        priority=req.priority,
        tags=req.tags,
        story_points=req.story_points,
        project_members=members,
    )

    response = AssignResponse(
        task_id=result["task_id"],
        suggestions=[AssigneeSuggestion(**s) for s in result["suggestions"]],
        model_version=result["model_version"],
    )

    # Log to JSON file
    top_conf = response.suggestions[0].confidence if response.suggestions else 0.0
    _log_request(
        task_id=req.task_id,
        num_suggestions=len(response.suggestions),
        top_confidence=top_conf,
        model_version=response.model_version,
    )

    payload = response.model_dump_json().encode("utf-8")
    return StreamingResponse(io.BytesIO(payload), media_type="application/json")
