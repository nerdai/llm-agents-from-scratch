"""LLM Agent Builder."""

from typing_extensions import Self

from llm_agents_from_scratch.agent.templates import (
    LLMAgentTemplates,
    default_templates,
)
from llm_agents_from_scratch.base import LLM
from llm_agents_from_scratch.base.tool import Tool
from llm_agents_from_scratch.tools.mcp import MCPToolProvider

from .llm_agent import LLMAgent


class LLMAgentBuilder:
    """A builder for LLM Agents."""

    def __init__(
        self,
        llm: LLM | None = None,
        tools: list[Tool] | None = None,
        templates: LLMAgentTemplates = default_templates,
        mcp_providers: list[MCPToolProvider] | None = None,
    ) -> None:
        """Initialize an LLMAgentBuilder."""
        self.llm = llm
        self.templates = templates
        self.mcp_providers = mcp_providers or []
        self.tools = tools or []

    def with_llm(self, llm: LLM) -> Self:
        """Set llm of builder."""
        self.llm = llm
        return self

    def with_tool(self, tool: Tool) -> Self:
        """Add tool to builder."""
        self.tools.append(tool)
        return self

    def with_tools(self, tools: list[Tool]) -> Self:
        """Add tools to builder."""
        self.tools.extend(tools)
        return self

    def with_templates(self, templates: LLMAgentTemplates) -> Self:
        """Set templates of builder."""
        self.templates = templates
        return self

    def with_mcp_provider(self, provider: MCPToolProvider) -> Self:
        """Add mcp provider to builder."""
        self.mcp_providers.append(provider)
        return self

    def with_mcp_providers(self, providers: list[MCPToolProvider]) -> Self:
        """Add mcp providers to builder."""
        self.mcp_providers.extend(providers)
        return self

    async def build(self) -> LLMAgent:
        """Build an LLMAgent."""
        raise NotImplementedError
