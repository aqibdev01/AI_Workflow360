from pydantic import BaseModel, Field


class DecomposeRequest(BaseModel):
    task_id: str
    title: str
    description: str = ""
    priority: str = "medium"
    project_context: str = ""
    existing_tags: list[str] = Field(default_factory=list)


class SubtaskSuggestion(BaseModel):
    title: str
    description: str
    priority: str = "medium"
    story_points: int = 2
    estimated_days: float = 1.0
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class DecomposeResponse(BaseModel):
    task_id: str
    subtasks: list[SubtaskSuggestion]
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    model_version: str
