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

    def __str__(self) -> str:
        """Return a prompt-ready XML string representation of the episode."""
        ts = self.completed_at.strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"  <episode>\n"
            f"    <task>{self.task.instruction}</task>\n"
            f"    <result>{self.result.content}\n    </result>\n"
            f"    <completed_at>{ts}</completed_at>\n"
            f"  </episode>"
        )
