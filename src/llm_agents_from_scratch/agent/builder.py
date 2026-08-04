"""LLM Agent Builder."""

import asyncio
from itertools import chain
from typing import TYPE_CHECKING

from typing_extensions import Self

from llm_agents_from_scratch.agent.templates import (
    LLMAgentTemplates,
    default_templates,
)
from llm_agents_from_scratch.base import LLM
from llm_agents_from_scratch.base.tool import Tool
from llm_agents_from_scratch.errors import LLMAgentBuilderError
from llm_agents_from_scratch.memory.memory import Memory
from llm_agents_from_scratch.tools import MCPTool
from llm_agents_from_scratch.tools.mcp import MCPToolProvider

from .llm_agent import LLMAgent

if TYPE_CHECKING:
    from llm_agents_from_scratch.subagents.spec import SubAgentSpec


class LLMAgentBuilder:
    """A builder for LLM Agents.

    Attributes:
        llm (LLM | None): The backbone LLM for the agent.
        tools (list[Tool]): Tools to equip the agent with.
        templates (LLMAgentTemplates): Prompt templates for the agent.
        mcp_providers (list[MCPToolProvider]): MCP providers for tool
            discovery.
        memories (list[Memory]): Memory backends for the agent.
        subagents (list[SubAgentSpec]): Sub-agent specs for the agent.
            Added in Chapter 9.
    """

    def __init__(  # noqa: PLR0913
        self,
        llm: LLM | None = None,
        tools: list[Tool] | None = None,
        templates: LLMAgentTemplates = default_templates,
        mcp_providers: list[MCPToolProvider] | None = None,
        # added in ch07
        memories: list[Memory] | None = None,
        # added in ch09
        subagents: "list[SubAgentSpec] | None" = None,
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
                    .with_memory(my_memory)
                    .with_subagent(spec)
                    .build()
                )

            Direct params::

                agent = await LLMAgentBuilder(
                    llm=llm,
                    tools=[my_tool],
                    mcp_providers=[provider],
                    memories=[my_memory],
                    subagents=[spec],
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
            memories (list[Memory] | None, optional): Memory backends
                for the agent. No default implementation is provided — the
                caller must supply a concrete subclass. Defaults to None.
            subagents (list[SubAgentSpec] | None, optional): Sub-agent specs
                to register on the agent. Defaults to None. Added in
                Chapter 9.
        """
        self.llm = llm
        self.templates = templates
        self.mcp_providers = mcp_providers or []
        self.tools = tools or []
        # added in ch07
        self.memories: list[Memory] = memories or []
        # added in ch09
        self.subagents: list[SubAgentSpec] = subagents or []

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

    def with_memory(self, memory: Memory) -> Self:
        """Add a memory backend to builder. Added in Chapter 7."""
        self.memories.append(memory)
        return self

    def with_memories(self, memories: list[Memory]) -> Self:
        """Add memory backends to builder. Added in Chapter 7."""
        self.memories.extend(memories)
        return self

    def with_subagent(self, spec: "SubAgentSpec") -> Self:
        """Add a sub-agent spec to builder. Added in Chapter 9.

        Args:
            spec (SubAgentSpec): The sub-agent spec to add.
        """
        self.subagents.append(spec)
        return self

    def with_subagents(self, specs: "list[SubAgentSpec]") -> Self:
        """Add sub-agent specs to builder. Added in Chapter 9.

        Args:
            specs (list[SubAgentSpec]): The sub-agent specs to add.
        """
        self.subagents.extend(specs)
        return self

    def with_default_subagents(self, llm: LLM | None = None) -> Self:
        """Add the default `general` + `explore` subagent specs.

        Added in Chapter 9.

        Args:
            llm (LLM | None, optional): Backbone LLM for both default
                subagents. Defaults to None, which inherits the
                builder's own `llm`.

        Raises:
            LLMAgentBuilderError: If neither `llm` nor the builder's
                own `llm` is set.
        """
        resolved_llm = llm or self.llm
        if not resolved_llm:
            raise LLMAgentBuilderError(
                "`llm` must be set on the builder or passed explicitly "
                "to `with_default_subagents()`",
            )

        # deferred import: subagents/recipes.py imports LLMAgent, which
        # would cycle back to this module at import time  # noqa: PLC0415
        from llm_agents_from_scratch.subagents.recipes import (  # noqa: PLC0415
            explore_subagent,
            general_subagent,
        )

        return self.with_subagents(
            [
                general_subagent(resolved_llm),
                explore_subagent(resolved_llm),
            ],
        )

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
            memories=self.memories,  # added in ch07
            subagents=self.subagents,  # added in ch09
        )
