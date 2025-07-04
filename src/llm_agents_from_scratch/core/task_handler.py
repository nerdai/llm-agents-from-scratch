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
from llm_agents_from_scratch.errors import TaskHandlerError

DEFAULT_GET_NEXT_INSTRUCTION_PROMPT = "{current_rollout}"


class TaskHandler(asyncio.Future):
    """Handler for processing tasks.

    Attributes:
        task: The task to execute.
        llm: The backbone LLM.
        tools: The tools the LLM agent can use.
        rollout: The execution log of the task.
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
        self.rollout = ""
        self._background_task: asyncio.Task | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def background_task(self) -> asyncio.Task:
        """Get the background ~asyncio.Task for the handler."""
        if not self._background_task:
            raise TaskHandlerError(
                "No background task is running for this handler.",
            )
        return self._background_task

    @background_task.setter
    def background_task(self, asyncio_task: asyncio.Task) -> None:
        """Setter for background_task."""
        if self._background_task is not None:
            raise TaskHandlerError("A background task has already been set.")
        self._background_task = asyncio_task

    async def get_next_step(self) -> TaskStep:
        """Based on task progress, determine next step.

        Returns:
            TaskStep: The next step to run, if `None` then Task is done.
        """
        async with self._lock:
            rollout = self.rollout

        if rollout == "":
            return TaskStep(instruction=self.task.instruction)

        prompt = DEFAULT_GET_NEXT_INSTRUCTION_PROMPT.format(
            current_rollout=rollout,
        )
        try:
            task_step = await self.llm.structured_output(
                prompt=prompt,
                mdl=TaskStep,
            )
        except Exception as e:
            raise TaskHandlerError(f"Failed to get next step: {str(e)}") from e

        return task_step

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
