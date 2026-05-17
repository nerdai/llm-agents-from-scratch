"""Data structures for episodic memory."""

from datetime import datetime

from pydantic import BaseModel, Field

from .agent import Task, TaskResult


class Episode(BaseModel):
    """A completed task to be stored in memory."""

    task: Task
    rollout: str
    result: TaskResult
    completed_at: datetime = Field(default_factory=datetime.now)
