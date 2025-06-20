"""Data Structures for LLM Agent."""

from pydantic import BaseModel


class Task(BaseModel):
    instruction: str


class TaskResult(BaseModel):
    response: str
    rollout: str
    error: bool = False
