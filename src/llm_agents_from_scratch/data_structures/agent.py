"""Data Structures for LLM Agent."""

from pydantic import BaseModel


class Task(BaseModel):
    instruction: str


class TaskStep(BaseModel):
    instruction: str


class TaskStepResult(BaseModel):
    task_step: TaskStep
    content: str | None
    last_step: bool = False


class TaskResult(BaseModel):
    task: Task
    content: str
    rollout: str
    error: bool = False
