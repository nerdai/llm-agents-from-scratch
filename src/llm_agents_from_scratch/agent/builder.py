"""LLM Agent Builder."""

import asyncio
from itertools import chain

from typing_extensions import Self

from llm_agents_from_scratch.agent.templates import (
    LLMAgentTemplates,
    default_templates,
)
from llm_agents_from_scratch.base import LLM
from llm_agents_from_scratch.base.tool import Tool
from llm_agents_from_scratch.errors import LLMAgentBuilderError
from llm_agents_from_scratch.tools import MCPTool
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
        """Initialize an LLMAgentBuilder.

        All parameters can also be set via fluent `with_*` builder methods for
        chained configuration.

        Examples:
            Fluent style::

                agent = await (
                    LLMAgentBuilder()
                    .with_llm(llm)
                    .with_tool(my_tool)
                    .with_mcp_provider(provider)
                    .build()
                )

            Direct params::

                agent = await LLMAgentBuilder(
                    llm=llm,
                    tools=[my_tool],
                    mcp_providers=[provider],
                ).build()

        Args:
            llm (LLM | None, optional): The backbone LLM for the agent.
                Required before calling `build()`. Defaults to None.
            tools (list[Tool] | None, optional): Initial list of tools to
                equip the agent with. Defaults to None.
            templates (LLMAgentTemplates, optional): Prompt templates for
                the agent. Defaults to `default_templates`.
            mcp_providers (list[MCPToolProvider] | None, optional): MCP
                providers for tool discovery. Tools are fetched during
                `build()`. Defaults to None.
        """
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
        """Build an LLMAgent with configured tools and MCP providers.

        Discovers tools from all registered MCP providers concurrently,
        combines them with manually added tools, and returns a configured
        LLMAgent.

        This is the recommended pattern for building agents with MCP tools.
        Alternatively, you can manually discover tools and pass them directly:

            provider = MCPToolProvider(name="github", url="...")
            tools = await provider.get_tools()
            agent = LLMAgent(llm=llm, tools=tools)

        Returns:
            LLMAgent: The configured agent with all tools.

        Raises:
            LLMAgentBuilderError: If `llm` is not set.
        """
        if not self.llm:
            raise LLMAgentBuilderError("`llm` must be set")

        # discover tools for mcp providers
        coros = []
        for provider in self.mcp_providers:
            coro = provider.get_tools()
            coros.append(coro)

        discovered_tools: list[list[MCPTool]] = await asyncio.gather(*coros)
        mcp_tools = list(chain.from_iterable(discovered_tools))

        return LLMAgent(
            llm=self.llm,
            tools=self.tools + mcp_tools,
            templates=self.templates,
        )
