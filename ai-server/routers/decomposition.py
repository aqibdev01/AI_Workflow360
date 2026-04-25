"""POST /api/decompose — Task decomposition endpoint (M6).
GET  /api/decompose/{task_id}/history — Past decompositions for a task.

Logs every request to a JSON log file for audit and model evaluation.
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
import io

from schemas.decomposition import DecomposeRequest, DecomposeResponse, SubtaskSuggestion
from models.decomposition.inference import decompose
from utils.auth import verify_api_key

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# In-memory request log (also persisted to JSON file)
_history: dict[str, list[dict]] = defaultdict(list)

# JSON log file for decomposition requests
_LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
_LOG_FILE = _LOG_DIR / "decomposition_requests.jsonl"


def _log_request(task_id: str, num_subtasks: int, confidence: float, model_version: str):
    """Append a JSON log entry for this decomposition request."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "num_subtasks": num_subtasks,
            "overall_confidence": confidence,
            "model_version": model_version,
        }
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("Failed to write decomposition log: %s", exc)


@router.post(
    "/decompose",
    summary="Decompose a task into subtasks",
    description=(
        "Takes a task's title, description, and priority, then uses the "
        "FLAN-T5 model (or mock) to suggest subtasks with story points, "
        "estimated days, and tags."
    ),
)
async def decompose_task(
    req: DecomposeRequest,
    _key: str = Depends(verify_api_key),
):
    # Warn if description is missing but proceed
    if not req.description:
        log.warning(
            "Decompose request for task %s has no description — results may be less accurate",
            req.task_id,
        )

    log.info("Decompose request for task %s: %s", req.task_id, req.title)

    result = decompose(
        task_id=req.task_id,
        title=req.title,
        description=req.description,
        priority=req.priority,
        project_context=req.project_context,
        existing_tags=req.existing_tags,
    )

    # Build response
    response = DecomposeResponse(
        task_id=result["task_id"],
        subtasks=[SubtaskSuggestion(**s) for s in result["subtasks"]],
        overall_confidence=result["overall_confidence"],
        model_version=result["model_version"],
    )

    # Log to JSON file
    _log_request(
        task_id=req.task_id,
        num_subtasks=len(response.subtasks),
        confidence=response.overall_confidence,
        model_version=response.model_version,
    )

    # Store in memory history
    _history[req.task_id].append(result)

    # Return as StreamingResponse (chunked) — HF Space's proxy mishandles
    # fixed-Content-Length responses and truncates the body mid-TLS.
    payload = response.model_dump_json().encode("utf-8")
    return StreamingResponse(
        io.BytesIO(payload),
        media_type="application/json",
    )


@router.get(
    "/decompose/{task_id}/history",
    summary="Get past decompositions for a task",
    description="Returns all previous decomposition results stored in memory for this server session.",
)
async def decompose_history(
    task_id: str,
    _key: str = Depends(verify_api_key),
):
    entries = _history.get(task_id, [])
    return {"task_id": task_id, "count": len(entries), "history": entries}
