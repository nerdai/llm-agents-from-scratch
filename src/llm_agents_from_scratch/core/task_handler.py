"""Task Handler"""

import asyncio
from typing import Any

from llm_agents_from_scratch.data_structures import (
    Task,
    TaskStep,
    TaskStepResult,
)


class TaskHandler(asyncio.Future):
    def __init__(self, task: Task, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.task = task
        self._asyncio_tasks: list[asyncio.Task] = []

    def add_asyncio_task(self, asyncio_task: asyncio.Task) -> None:
        self._asyncio_tasks.append(asyncio_task)

    async def get_next_step(self) -> TaskStep | None:
        """Based on task progress, determine next step.

        Returns:
            TaskStep | None: The next step to run, if `None` then Task is done.
        """
        pass

    async def run_step(self, step: TaskStep) -> TaskStepResult:
        """Run next step of a given task.

        Example: perform tool call, generated LLM response, etc.

        Args:
            last_step (Any): The result of the previous step.

        Returns:
            Any: The result of the next sub step and sets result if Task the completion
            of the sub-step represents the completion of the Task.
        """
        pass
