"""Data structures for episodic memory."""

from datetime import datetime

from pydantic import BaseModel

from .agent import Task, TaskResult
from .llm import ChatMessage


class Episode(BaseModel):
    """A completed task to be stored in memory."""

    task: Task
    rollout: list[ChatMessage]
    result: TaskResult
    completed_at: datetime
