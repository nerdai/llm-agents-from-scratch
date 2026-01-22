"""Unit tests for MCPTool."""

from unittest.mock import MagicMock

from llm_agents_from_scratch.tools.mcp import MCPTool


def test_function_tool_init() -> None:
    """Test MCPTool initialization."""
    tool = MCPTool(
        provider=MagicMock(),
        name="mock tool",
        desc="mock desc",
        params_json_schema={"param1": {"type": "number"}},
    )

    assert tool.name == "mock tool"
    assert tool.description == "mock desc"
    assert tool.additional_annotations is None
    assert tool.parameters_json_schema == {"param1": {"type": "number"}}


def test_tool_call() -> None:
    """Tests MCP tool call."""
    ...
