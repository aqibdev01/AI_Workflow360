from pydantic import BaseModel, Field


class MemberProfile(BaseModel):
    model_config = {"extra": "ignore"}

    user_id: str
    role: str
    skills: list[str] = Field(default_factory=list)
    current_task_count: int = 0
    current_story_points: int = 0
    completed_tasks_last_30d: int = 0


class AssignRequest(BaseModel):
    task_id: str
    title: str
    description: str = ""
    priority: str = "medium"
    tags: list[str] = Field(default_factory=list)
    story_points: int | None = None
    project_members: list[MemberProfile] = Field(default_factory=list)


class AssigneeSuggestion(BaseModel):
    user_id: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    scoring_breakdown: dict = Field(default_factory=dict)


class AssignResponse(BaseModel):
    task_id: str
    suggestions: list[AssigneeSuggestion]
    model_version: str
