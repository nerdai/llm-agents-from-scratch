"""Unit tests for LLMAgentBuilder."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.types import AgentCard, AgentInterface

from llm_agents_from_scratch import LLMAgentBuilder
from llm_agents_from_scratch.a2a import A2AAgentSpec
from llm_agents_from_scratch.agent import LLMAgent
from llm_agents_from_scratch.agent.templates import default_templates
from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures import Task
from llm_agents_from_scratch.errors import LLMAgentBuilderError, LLMAgentError
from llm_agents_from_scratch.memory.memory import Memory
from llm_agents_from_scratch.subagents import SubAgentSpec, UseSubAgentTool
from llm_agents_from_scratch.tools.mcp.tool import MCPTool


def test_init() -> None:
    """Tests init of builder with different patterns."""
    mock_llm = MagicMock()
    mock_tool = MagicMock()
    mock_mcp_provider = MagicMock()
    mock_memory = MagicMock(spec=Memory)

    # direct params
    builder = LLMAgentBuilder(
        llm=mock_llm,
        tools=[mock_tool],
        mcp_providers=[mock_mcp_provider],
        memories=[mock_memory],
    )
    assert builder.llm == mock_llm
    assert builder.tools == [mock_tool]
    assert builder.mcp_providers == [mock_mcp_provider]
    assert builder.templates == default_templates
    assert builder.memories == [mock_memory]

    # fluent chaining
    builder = (
        LLMAgentBuilder()
        .with_tool(mock_tool)
        .with_llm(mock_llm)
        .with_templates(default_templates)
        .with_mcp_provider(mock_mcp_provider)
        .with_memory(mock_memory)
    )
    assert builder.llm == mock_llm
    assert builder.tools == [mock_tool]
    assert builder.mcp_providers == [mock_mcp_provider]
    assert builder.templates == default_templates
    assert builder.memories == [mock_memory]

    # mix
    builder = (
        LLMAgentBuilder(llm=mock_llm)
        .with_tools([mock_tool])
        .with_mcp_providers([mock_mcp_provider])
        .with_memories([mock_memory])
    )
    assert builder.llm == mock_llm
    assert builder.tools == [mock_tool]
    assert builder.mcp_providers == [mock_mcp_provider]
    assert builder.templates == default_templates
    assert builder.memories == [mock_memory]


@pytest.mark.asyncio
async def test_build() -> None:
    """Tests build for LLMAgent."""
    mock_llm = MagicMock()
    mock_tool = MagicMock()
    mock_mcp_provider = MagicMock()
    mock_get_tools = AsyncMock()
    test_mcp_tool = MCPTool(
        provider=mock_mcp_provider,
        name="mock_provider.mock_mcp_tool",
        desc="mock desc",
        params_json_schema={"param1": {"type": "number"}},
    )
    mock_get_tools.return_value = [test_mcp_tool]
    mock_mcp_provider.get_tools = mock_get_tools

    # direct params
    builder = LLMAgentBuilder(
        llm=mock_llm,
        tools=[mock_tool],
        mcp_providers=[mock_mcp_provider],
    )
    agent = await builder.build()

    assert set(agent.tools) == {test_mcp_tool, mock_tool}
    mock_get_tools.assert_awaited_once()
    assert agent.llm == builder.llm
    assert agent.templates == builder.templates
    assert agent.memories == []


@pytest.mark.asyncio
async def test_build_passes_memories_to_agent() -> None:
    """Tests build passes memories through to the constructed LLMAgent."""
    mock_llm = MagicMock()
    mock_memory_a = MagicMock(spec=Memory)
    mock_memory_b = MagicMock(spec=Memory)

    agent = await (
        LLMAgentBuilder(llm=mock_llm)
        .with_memory(mock_memory_a)
        .with_memory(mock_memory_b)
        .build()
    )

    assert agent.memories == [mock_memory_a, mock_memory_b]


@pytest.mark.asyncio
async def test_build_raises_error_with_no_llm_set() -> None:
    """Tests build for LLMAgent."""
    mock_tool = MagicMock()

    with pytest.raises(LLMAgentBuilderError, match="`llm` must be set"):
        await LLMAgentBuilder().with_tool(mock_tool).build()


# ---------------------------------------------------------------------------
# Subagents tests (Chapter 9)
# ---------------------------------------------------------------------------


def test_init_with_subagents(mock_llm: BaseLLM) -> None:
    """Tests builder init stores subagents list."""
    spec = SubAgentSpec(
        name="researcher",
        description="Looks things up.",
        agent=LLMAgent(llm=mock_llm),
    )
    builder = LLMAgentBuilder(subagents=[spec])

    assert builder.subagents == [spec]


def test_with_subagent_fluent(mock_llm: BaseLLM) -> None:
    """Tests with_subagent() appends a spec and returns self."""
    spec = SubAgentSpec(
        name="coder",
        description="Writes code.",
        agent=LLMAgent(llm=mock_llm),
    )
    builder = LLMAgentBuilder().with_subagent(spec)

    assert builder.subagents == [spec]


def test_with_subagents_fluent(mock_llm: BaseLLM) -> None:
    """Tests with_subagents() extends specs and returns self."""
    spec_a = SubAgentSpec(
        name="researcher",
        description="Looks things up.",
        agent=LLMAgent(llm=mock_llm),
    )
    spec_b = SubAgentSpec(
        name="coder",
        description="Writes code.",
        agent=LLMAgent(llm=mock_llm),
    )
    builder = LLMAgentBuilder().with_subagents([spec_a, spec_b])

    assert builder.subagents == [spec_a, spec_b]


@pytest.mark.asyncio
async def test_build_passes_subagents_to_agent(mock_llm: BaseLLM) -> None:
    """Tests build() wires subagents registry into LLMAgent."""
    spec = SubAgentSpec(
        name="researcher",
        description="Looks things up.",
        agent=LLMAgent(llm=mock_llm),
    )
    agent = await LLMAgentBuilder(llm=mock_llm).with_subagent(spec).build()

    assert "researcher" in agent.subagents_registry
    assert agent.subagents_registry["researcher"] is spec


@pytest.mark.asyncio
async def test_build_use_subagent_tool_present(mock_llm: BaseLLM) -> None:
    """Tests build() with subagents makes UseSubAgentTool available in run."""
    spec = SubAgentSpec(
        name="coder",
        description="Writes code.",
        agent=LLMAgent(llm=mock_llm),
    )
    agent = await LLMAgentBuilder(llm=mock_llm).with_subagent(spec).build()

    handler = LLMAgent.TaskHandler(
        llm_agent=agent,
        task=Task(instruction="test"),
    )
    assert isinstance(handler._use_subagent_tool, UseSubAgentTool)


@pytest.mark.asyncio
async def test_build_raises_on_duplicate_subagent_names(
    mock_llm: BaseLLM,
) -> None:
    """Tests build() raises LLMAgentBuilderError on duplicate subagent names."""
    spec_a = SubAgentSpec(
        name="researcher",
        description="First.",
        agent=LLMAgent(llm=mock_llm),
    )
    spec_b = SubAgentSpec(
        name="researcher",
        description="Second.",
        agent=LLMAgent(llm=mock_llm),
    )
    with pytest.raises(LLMAgentError, match="duplicate"):
        await (
            LLMAgentBuilder(llm=mock_llm)
            .with_subagents([spec_a, spec_b])
            .build()
        )


# ---------------------------------------------------------------------------
# A2A tests (Chapter 10)
# ---------------------------------------------------------------------------


def _a2a_spec(name: str) -> A2AAgentSpec:
    card = AgentCard(
        name=name,
        description="A peer agent.",
        supported_interfaces=[AgentInterface(url="http://peer:9999")],
    )
    return A2AAgentSpec.from_agent_card(agent_card=card)


def test_init_with_a2a_agents() -> None:
    """Tests builder init stores a2a_agents list."""
    spec = _a2a_spec("researcher")
    builder = LLMAgentBuilder(a2a_agents=[spec])

    assert builder.a2a_agents == [spec]


def test_with_a2a_agent_fluent() -> None:
    """Tests with_a2a_agent() appends a spec and returns self."""
    spec = _a2a_spec("coder")
    builder = LLMAgentBuilder().with_a2a_agent(spec)

    assert builder.a2a_agents == [spec]


def test_with_a2a_agents_fluent() -> None:
    """Tests with_a2a_agents() extends specs and returns self."""
    spec_a = _a2a_spec("researcher")
    spec_b = _a2a_spec("coder")
    builder = LLMAgentBuilder().with_a2a_agents([spec_a, spec_b])

    assert builder.a2a_agents == [spec_a, spec_b]


@pytest.mark.asyncio
async def test_build_passes_a2a_agents_to_agent(mock_llm: BaseLLM) -> None:
    """Tests build() wires a2a_agents registry into LLMAgent."""
    spec = _a2a_spec("researcher")
    agent = await LLMAgentBuilder(llm=mock_llm).with_a2a_agent(spec).build()

    assert "researcher" in agent.a2a_agents_registry
    assert agent.a2a_agents_registry["researcher"] is spec


@pytest.mark.asyncio
async def test_build_raises_on_duplicate_a2a_agent_names(
    mock_llm: BaseLLM,
) -> None:
    """Tests build() raises LLMAgentError on duplicate a2a_agent names."""
    spec_a = _a2a_spec("researcher")
    spec_b = _a2a_spec("researcher")
    with pytest.raises(LLMAgentError, match="duplicate"):
        await (
            LLMAgentBuilder(llm=mock_llm)
            .with_a2a_agents([spec_a, spec_b])
            .build()
        )
