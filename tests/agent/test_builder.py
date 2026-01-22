"""Unit tests for LLMAgentBuilder."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_agents_from_scratch import LLMAgentBuilder
from llm_agents_from_scratch.agent.templates import default_templates
from llm_agents_from_scratch.tools.mcp.tool import MCPTool


def test_init() -> None:
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
