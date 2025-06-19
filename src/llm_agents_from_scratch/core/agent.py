"""Agent Module."""

from typing_extensions import Self

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import BaseTool


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
