"""Agent Module."""

import asyncio

from typing_extensions import Self

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import AsyncBaseTool, BaseTool
from llm_agents_from_scratch.data_structures import (
    Task,
    TaskHandlerResult,
    TaskResult,
    TaskStep,
)
from llm_agents_from_scratch.errors import LLMAgentError
from llm_agents_from_scratch.logger import get_logger

from .task_handler import TaskHandler


class LLMAgent:
    """A simple LLM Agent Class.

    Attributes:
        llm: The backbone LLM
        tools_registry: The tools the LLM agent can use represented as a dict.
        logger: LLMAgent logger.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool | AsyncBaseTool] | None = None,
    ):
        """Initialize an LLMAgent.

        Args:
            llm (BaseLLM): The backbone LLM of the LLM agent.
            tools (list[BaseTool], optional): The set of tools for the LLM
                agent. Defaults to None.

        """
        self.llm = llm
        tools = tools or []
        # validate no duplications in tool names
        if len({t.name for t in tools}) < len(tools):
            raise LLMAgentError(
                "Provided tool list contains duplicate tool names.",
            )
        self.tools_registry = {t.name: t for t in tools}
        self.logger = get_logger(self.__class__.__name__)

    @property
    def tools(self) -> list[BaseTool | AsyncBaseTool]:
        """Return tools as list."""
        return list(self.tools_registry.values())

    def add_tool(self, tool: BaseTool | AsyncBaseTool) -> Self:
        """Add a tool to the agents tool set.

        NOTE: Supports fluent style for convenience.

        Args:
            tool (BaseTool | AsyncBaseTool): The tool to equip the LLM agent.

        """
        if tool.name in self.tools_registry:
            raise LLMAgentError(f"Tool with name {tool.name} already exists.")
        self.tools_registry[tool.name] = tool
        return self

    def run(self, task: Task) -> TaskHandler:
        """Agent's processing loop for executing tasks.

        Asynchronously run `task`.
        """
        task_handler = TaskHandler(task, self.llm, self.tools)

        async def _run() -> None:
            """Asynchronously process the task."""
            self.logger.info(f"🚀 Starting task: {task.instruction}")
            step_result = None
            while not task_handler.done():
                try:
                    next_step = await task_handler.get_next_step(step_result)

                    match next_step:
                        case TaskStep():
                            step_result = await task_handler.run_step(
                                next_step,
                            )
                        case TaskResult():
                            async with task_handler._lock:
                                rollout = task_handler.rollout

                            task_handler.set_result(
                                TaskHandlerResult(
                                    task_result=next_step,
                                    rollout=rollout,
                                ),
                            )
                            self.logger.info(
                                f"🏁 Task completed: {next_step.content}",
                            )

                except Exception as e:
                    task_handler.set_exception(e)

        task_handler.background_task = asyncio.create_task(_run())

        return task_handler
