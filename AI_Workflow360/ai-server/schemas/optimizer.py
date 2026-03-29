from pydantic import BaseModel, Field


class SprintTask(BaseModel):
    task_id: str
    title: str
    status: str
    priority: str = "medium"
    assignee_id: str | None = None
    story_points: int | None = None
    due_date: str | None = None
    created_at: str = ""


class SprintAnalysisRequest(BaseModel):
    sprint_id: str
    sprint_name: str
    start_date: str
    end_date: str
    capacity: float | None = None
    tasks: list[SprintTask] = Field(default_factory=list)
    member_workloads: dict[str, int] = Field(default_factory=dict)


class Bottleneck(BaseModel):
    type: str
    description: str
    affected_task_ids: list[str] = Field(default_factory=list)
    severity: str = "medium"


class Recommendation(BaseModel):
    action: str
    reason: str
    priority: str = "medium"


class SprintAnalysisResponse(BaseModel):
    sprint_id: str
    risk_level: str
    risk_score: float = Field(0.0, ge=0.0, le=1.0)
    bottlenecks: list[Bottleneck] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    model_version: str
