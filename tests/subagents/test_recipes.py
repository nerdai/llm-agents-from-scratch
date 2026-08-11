"""Unit tests for default subagent recipes."""

from unittest.mock import MagicMock

import pytest

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.memory.memory import Memory
from llm_agents_from_scratch.subagents.constants import (
    EXPLORE_MAX_STEPS,
    EXPLORE_NAME,
    GENERAL_MAX_STEPS,
    GENERAL_NAME,
)
from llm_agents_from_scratch.subagents.recipes import (
    explore_subagent,
    general_subagent,
)

CUSTOM_MAX_STEPS_GENERAL = 5
CUSTOM_MAX_STEPS_EXPLORE = 3


def test_general_subagent_defaults(mock_llm: BaseLLM) -> None:
    """Tests general_subagent applies default name and max_steps."""
    spec = general_subagent(mock_llm)

    assert spec.name == GENERAL_NAME
    assert spec.max_steps == GENERAL_MAX_STEPS


@pytest.mark.asyncio
async def test_general_subagent_equips_default_tools(mock_llm: BaseLLM) -> None:
    """Tests general_subagent equips the agent with DEFAULT_TOOLS."""
    spec = general_subagent(mock_llm)
    agent = await spec.builder.build()

    tool_names = {tool.name for tool in agent.tools_registry.values()}
    assert "from_scratch__read_file" in tool_names
    assert "from_scratch__python_interpreter" in tool_names


def test_general_subagent_custom_name_and_max_steps(
    mock_llm: BaseLLM,
) -> None:
    """Tests general_subagent accepts custom name and max_steps."""
    spec = general_subagent(
        mock_llm,
        name="researcher",
        max_steps=CUSTOM_MAX_STEPS_GENERAL,
    )

    assert spec.name == "researcher"
    assert spec.max_steps == CUSTOM_MAX_STEPS_GENERAL


def test_general_subagent_passes_memories(mock_llm: BaseLLM) -> None:
    """Tests general_subagent forwards memories to the builder."""
    memory = MagicMock(spec=Memory)
    spec = general_subagent(mock_llm, memories=[memory])

    assert spec.builder.memories == [memory]


def test_explore_subagent_defaults(mock_llm: BaseLLM) -> None:
    """Tests explore_subagent applies default name and max_steps."""
    spec = explore_subagent(mock_llm)

    assert spec.name == EXPLORE_NAME
    assert spec.max_steps == EXPLORE_MAX_STEPS


@pytest.mark.asyncio
async def test_explore_subagent_equips_read_file_tool_only(
    mock_llm: BaseLLM,
) -> None:
    """Tests explore_subagent equips only ReadFileTool."""
    spec = explore_subagent(mock_llm)
    agent = await spec.builder.build()

    tool_names = {tool.name for tool in agent.tools_registry.values()}
    assert tool_names == {"from_scratch__read_file"}


def test_explore_subagent_custom_name_and_max_steps(
    mock_llm: BaseLLM,
) -> None:
    """Tests explore_subagent accepts custom name and max_steps."""
    spec = explore_subagent(
        mock_llm,
        name="scout",
        max_steps=CUSTOM_MAX_STEPS_EXPLORE,
    )

    assert spec.name == "scout"
    assert spec.max_steps == CUSTOM_MAX_STEPS_EXPLORE


def test_explore_subagent_passes_memories(mock_llm: BaseLLM) -> None:
    """Tests explore_subagent forwards memories to the builder."""
    memory = MagicMock(spec=Memory)
    spec = explore_subagent(mock_llm, memories=[memory])

    assert spec.builder.memories == [memory]


def test_explore_max_steps_lower_than_general() -> None:
    """Tests explore's default max_steps is lower than general's."""
    assert EXPLORE_MAX_STEPS < GENERAL_MAX_STEPS
