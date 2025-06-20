"""Agent Module."""

from typing_extensions import Self

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import Task, TaskResult

from .task_handler import TaskHandler


class LLMAgent:
    """A simple LLM Agent Class."""

    def __init__(self, llm: BaseLLM, tools: list[BaseTool] = []):
        self.llm = llm
        self.tools = tools

    def add_tool(self, tool: BaseTool) -> Self:
        """Add a tool to the agents tool set.

        NOTE: Supports fluent style for convenience.

        Args:
            tool (BaseTool): The tool to equip the LLM agent.
        """
        self.tools = self.tools + [tool]
        return self

    async def run(task: Task) -> TaskResult:
        """Asynchronously run `task`."""
        task_handler = TaskHandler()
        steps = []
        while not task_handler.done():
            step_output = await task_handler.run_step(steps[-1])

        # prepare rollout
        rollout = ""
        if task_handler.exception():
            return TaskResult(response="Error", error=True, rollout=rollout)

        return TaskResult(response=str(task_handler.result()), rollout=rollout)
