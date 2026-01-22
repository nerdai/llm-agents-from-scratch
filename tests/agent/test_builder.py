"""Unit tests for LLMAgentBuilder."""

from unittest.mock import MagicMock

from llm_agents_from_scratch import LLMAgentBuilder
from llm_agents_from_scratch.agent.templates import default_templates


def test_init():
    """Tests init of builder with different patterns."""
    mock_llm = MagicMock()
    mock_tool = MagicMock()
    mock_mcp_provider = MagicMock()

    # direct params
    builder = LLMAgentBuilder(
        llm=mock_llm,
        tools=[mock_tool],
        mcp_providers=[mock_mcp_provider],
    )
    assert builder.llm == mock_llm
    assert builder.tools == [mock_tool]
    assert builder.mcp_providers == [mock_mcp_provider]
    assert builder.templates == default_templates

    # fluent chaining
    builder = (
        LLMAgentBuilder()
        .with_tool(mock_tool)
        .with_llm(mock_llm)
        .with_templates(default_templates)
        .with_mcp_provider(mock_mcp_provider)
    )
    assert builder.llm == mock_llm
    assert builder.tools == [mock_tool]
    assert builder.mcp_providers == [mock_mcp_provider]
    assert builder.templates == default_templates

    # mix
    builder = (
        LLMAgentBuilder(llm=mock_llm)
        .with_tools([mock_tool])
        .with_mcp_providers([mock_mcp_provider])
    )
    assert builder.llm == mock_llm
    assert builder.tools == [mock_tool]
    assert builder.mcp_providers == [mock_mcp_provider]
    assert builder.templates == default_templates
