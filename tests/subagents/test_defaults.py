"""Unit tests for default subagent specs."""

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.subagents.defaults import (
    EXPLORE_MAX_STEPS,
    GENERAL_MAX_STEPS,
    explore_subagent_spec,
    general_subagent_spec,
)

CUSTOM_MAX_STEPS_GENERAL = 5
CUSTOM_MAX_STEPS_EXPLORE = 3


def test_general_subagent_spec_defaults(mock_llm: BaseLLM) -> None:
    """Tests general_subagent_spec applies default name and max_steps."""
    spec = general_subagent_spec(mock_llm)

    assert spec.name == "general"
    assert spec.max_steps == GENERAL_MAX_STEPS


def test_general_subagent_spec_equips_default_tools(mock_llm: BaseLLM) -> None:
    """Tests general_subagent_spec equips the agent with DEFAULT_TOOLS."""
    spec = general_subagent_spec(mock_llm)

    tool_names = {tool.name for tool in spec.agent.tools_registry.values()}
    assert "from_scratch__read_file" in tool_names
    assert "from_scratch__python_interpreter" in tool_names


def test_general_subagent_spec_custom_name_and_max_steps(
    mock_llm: BaseLLM,
) -> None:
    """Tests general_subagent_spec accepts custom name and max_steps."""
    spec = general_subagent_spec(
        mock_llm,
        name="researcher",
        max_steps=CUSTOM_MAX_STEPS_GENERAL,
    )

    assert spec.name == "researcher"
    assert spec.max_steps == CUSTOM_MAX_STEPS_GENERAL


def test_explore_subagent_spec_defaults(mock_llm: BaseLLM) -> None:
    """Tests explore_subagent_spec applies default name and max_steps."""
    spec = explore_subagent_spec(mock_llm)

    assert spec.name == "explore"
    assert spec.max_steps == EXPLORE_MAX_STEPS


def test_explore_subagent_spec_equips_read_file_tool_only(
    mock_llm: BaseLLM,
) -> None:
    """Tests explore_subagent_spec equips only ReadFileTool."""
    spec = explore_subagent_spec(mock_llm)

    tool_names = {tool.name for tool in spec.agent.tools_registry.values()}
    assert tool_names == {"from_scratch__read_file"}


def test_explore_subagent_spec_custom_name_and_max_steps(
    mock_llm: BaseLLM,
) -> None:
    """Tests explore_subagent_spec accepts custom name and max_steps."""
    spec = explore_subagent_spec(
        mock_llm,
        name="scout",
        max_steps=CUSTOM_MAX_STEPS_EXPLORE,
    )

    assert spec.name == "scout"
    assert spec.max_steps == CUSTOM_MAX_STEPS_EXPLORE


def test_explore_max_steps_lower_than_general() -> None:
    """Tests explore's default max_steps is lower than general's."""
    assert EXPLORE_MAX_STEPS < GENERAL_MAX_STEPS
