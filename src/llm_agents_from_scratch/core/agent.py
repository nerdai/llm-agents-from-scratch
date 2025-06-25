"""Agent Module."""

import asyncio

from typing_extensions import Self

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import Task, TaskResult

from .task_handler import TaskHandler


class LLMAgent:
    """A simple LLM Agent Class."""

    def __init__(self, llm: BaseLLM, tools: list[BaseTool] | None = None):
        """Initialize an LLMAgent.

        Args:
            llm (BaseLLM): The backbone LLM of the LLM agent.
            tools (list[BaseTool], optional): The set of tools for the LLM
                agent. Defaults to None.

        """
        self.llm = llm
        self.tools = tools or []

    def add_tool(self, tool: BaseTool) -> Self:
        """Add a tool to the agents tool set.

        NOTE: Supports fluent style for convenience.

        Args:
            tool (BaseTool): The tool to equip the LLM agent.

        """
        self.tools = self.tools + [tool]
        return self

    def run(self, task: Task) -> TaskHandler:
        """Asynchronously run `task`."""
        task_handler = TaskHandler(task, self.llm, self.tools)

        async def _run() -> None:
            """Asynchronously process the task."""
            while not task_handler.done():
                try:
                    step = await task_handler.get_next_step()
                    step_result = await task_handler.run_step(step)
                    if step_result.last_step:
                        task_result = TaskResult(
                            task=task,
                            content=step_result.content,
                            rollout="",
                        )
                        task_handler.set_result(task_result)
                except Exception as e:
                    task_handler.set_exception(e)

        task_handler.add_asyncio_task(asyncio.create_task(_run()))

        return task_handler
