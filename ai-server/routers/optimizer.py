"""POST /api/analyze-sprint  — Sprint bottleneck analysis endpoint (M11).
POST /api/analyze-project — Batch analysis for all active sprints in a project.

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
from pydantic import BaseModel, Field

from schemas.optimizer import (
    SprintAnalysisRequest,
    SprintAnalysisResponse,
    Bottleneck,
    Recommendation,
)
from models.optimizer.inference import analyze, analyze_project
from utils.auth import verify_api_key

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# JSON log file
_LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
_LOG_FILE = _LOG_DIR / "sprint_analysis_requests.jsonl"


def _log_request(sprint_id: str, risk_level: str, risk_score: float, bottleneck_count: int, model_version: str):
    """Append a JSON log entry."""
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sprint_id": sprint_id,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "bottleneck_count": bottleneck_count,
            "model_version": model_version,
        }
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("Failed to write analysis log: %s", exc)


@router.post(
    "/analyze-sprint",
    summary="Analyze a sprint for bottleneck risks",
    description=(
        "Uses XGBoost classifier + rule-based detection to identify sprint risks. "
        "Returns risk level, bottleneck details, and actionable recommendations."
    ),
)
async def analyze_sprint(
    req: SprintAnalysisRequest,
    _key: str = Depends(verify_api_key),
):
    log.info(
        "Sprint analysis for %s (%d tasks)",
        req.sprint_id,
        len(req.tasks),
    )

    tasks = [t.model_dump() for t in req.tasks]

    result = analyze(
        sprint_id=req.sprint_id,
        sprint_name=req.sprint_name,
        start_date=req.start_date,
        end_date=req.end_date,
        capacity=req.capacity,
        tasks=tasks,
        member_workloads=req.member_workloads,
    )

    response = SprintAnalysisResponse(
        sprint_id=result["sprint_id"],
        risk_level=result["risk_level"],
        risk_score=result["risk_score"],
        bottlenecks=[Bottleneck(**b) for b in result["bottlenecks"]],
        recommendations=[Recommendation(**r) for r in result["recommendations"]],
        model_version=result["model_version"],
    )

    # Log
    _log_request(
        sprint_id=req.sprint_id,
        risk_level=response.risk_level,
        risk_score=response.risk_score,
        bottleneck_count=len(response.bottlenecks),
        model_version=response.model_version,
    )

    payload = response.model_dump_json().encode("utf-8")
    return StreamingResponse(io.BytesIO(payload), media_type="application/json")


# ---- Batch: analyze all sprints for a project ----

class ProjectAnalysisRequest(BaseModel):
    project_id: str
    sprints: list[SprintAnalysisRequest] = Field(default_factory=list)


class ProjectAnalysisResponse(BaseModel):
    project_id: str
    risk_level: str
    risk_score: float
    sprint_count: int
    bottleneck_count: int
    top_recommendations: list[Recommendation] = Field(default_factory=list)
    sprint_results: list[SprintAnalysisResponse] = Field(default_factory=list)
    model_version: str


@router.post(
    "/analyze-project",
    response_model=ProjectAnalysisResponse,
    summary="Analyze all active sprints in a project (batch)",
)
async def analyze_project_endpoint(
    req: ProjectAnalysisRequest,
    _key: str = Depends(verify_api_key),
) -> ProjectAnalysisResponse:
    log.info(
        "Project analysis for %s (%d sprints)",
        req.project_id,
        len(req.sprints),
    )

    sprint_results = []
    for sprint_req in req.sprints:
        tasks = [t.model_dump() for t in sprint_req.tasks]
        result = analyze(
            sprint_id=sprint_req.sprint_id,
            sprint_name=sprint_req.sprint_name,
            start_date=sprint_req.start_date,
            end_date=sprint_req.end_date,
            capacity=sprint_req.capacity,
            tasks=tasks,
            member_workloads=sprint_req.member_workloads,
        )
        sprint_results.append(result)

    summary = analyze_project(sprint_results)

    return ProjectAnalysisResponse(
        project_id=req.project_id,
        risk_level=summary["risk_level"],
        risk_score=summary["risk_score"],
        sprint_count=summary["sprint_count"],
        bottleneck_count=summary["bottleneck_count"],
        top_recommendations=[Recommendation(**r) for r in summary["top_recommendations"]],
        sprint_results=[
            SprintAnalysisResponse(
                sprint_id=r["sprint_id"],
                risk_level=r["risk_level"],
                risk_score=r["risk_score"],
                bottlenecks=[Bottleneck(**b) for b in r["bottlenecks"]],
                recommendations=[Recommendation(**rec) for rec in r["recommendations"]],
                model_version=r["model_version"],
            )
            for r in sprint_results
        ],
        model_version=summary["model_version"],
    )
