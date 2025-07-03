"""Task Handler."""

import asyncio
from typing import Any

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import (
    Task,
    TaskStep,
    TaskStepResult,
)


class TaskHandler(asyncio.Future):
    """Handler for processing tasks.

    Attributes:
        task: The task to execute.
        llm: The backbone LLM.
        tools: The tools the LLM agent can use.
    """

    def __init__(
        self,
        task: Task,
        llm: BaseLLM,
        tools: list[BaseTool],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize a TaskHandler.

        Args:
            task (Task): The task to process.
            llm (BaseLLM): The backbone LLM.
            tools (list[BaseTool]): The tools the LLM can use.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.task = task
        self.llm = llm
        self.tools = tools
        self._background_task: asyncio.Task | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def background_task(self) -> asyncio.Task:
        """Get the background ~asyncio.Task for the handler."""
        if not self._background_task:
            raise ValueError("No background task is running for this handler.")
        return self._background_task

    @background_task.setter
    def background_task(self, asyncio_task: asyncio.Task) -> None:
        """Setter for background_task."""
        if self._background_task is not None:
            raise ValueError("A background task has already been set.")
        self.background_task = asyncio_task

    async def get_next_step(self) -> TaskStep | None:
        """Based on task progress, determine next step.

        Returns:
            TaskStep | None: The next step to run, if `None` then Task is done.
        """
        # TODO: implement
        pass  # pragma: no cover

    async def run_step(self, step: TaskStep) -> TaskStepResult:
        """Run next step of a given task.

        Example: perform tool call, generated LLM response, etc.

        Args:
            step (TaskStep): The step to execute.

        Returns:
            TaskStepResult: The result of the step execution.
        """
        # TODO: implement
        pass  # pragma: no cover
