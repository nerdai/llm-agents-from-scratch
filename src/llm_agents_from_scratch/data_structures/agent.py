"""Data Structures for LLM Agent."""

import uuid

from pydantic import BaseModel, Field


class Task(BaseModel):
    """Represents a single task with an instruction.

    Attributes:
        instruction: The instruction for the task.
    """

    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str


class TaskStep(BaseModel):
    """Represents a step within a task and its own instruction.

    Attributes:
        instruction: The instruction for the task.
    """

    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    instruction: str = Field(
        description="The instruction for this step in the task.",
    )


class TaskStepResult(BaseModel):
    """The result of a task step execution.

    Attributes:
        task_step: The `TaskStep` that was executed.
        content: The content results of the execution.
        last_step: Whether or not the step was the final step for the parent
            Task.
    """

    task_step_id: str
    content: str


class TaskResult(BaseModel):
    """The result of the task execution.

    Attributes:
        task: The `Task` that was executed.
        content: The content results of the task execution.
        error: Whether or not the execution resulted in an error.
    """

    task_id: str
    content: str
    error: bool = False

    def __str__(self) -> str:
        """String representation of TaskResult."""
        return self.content


class GetNextStep(BaseModel):
    """Structured output for TaskHandler."""

    task_step: TaskStep | None = Field(
        description="If a next step is required. Otherwise is set to `None`.",
    )
    task_result: TaskResult | None = Field(
        description=(
            "If no next step is required, the task has a final result. "
            "Otherwise is set to `None`."
        ),
    )


class TaskHandlerResult(BaseModel):
    """Task Handler Future Result."""

    task_result: TaskResult
    rollout: str
