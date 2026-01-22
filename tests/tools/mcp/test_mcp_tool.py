"""Unit tests for MCPTool."""

from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import ClientSession, ListToolsResult, Tool
from mcp.types import CallToolResult, TextContent

from llm_agents_from_scratch.data_structures import ToolCall
from llm_agents_from_scratch.tools.mcp import MCPTool, MCPToolProvider


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


@pytest.fixture()
def mock_client_session() -> Callable[..., AsyncContextManager[AsyncMock]]:
    """Mock ClientSession."""
    client_session = AsyncMock(spec=ClientSession)
    mock_list_tools = AsyncMock()
    mock_list_tools.return_value = ListToolsResult(
        tools=[
            Tool(
                name="mock_tool",
                description="mock_desc",
                inputSchema={"param1": {"type": "number"}},
            ),
        ],
    )
    client_session.call_tool.side_effect = [
        CallToolResult(
            content=[
                TextContent(text="42", type="text"),
            ],
            isError=False,
        ),
        CallToolResult(
            content=[
                TextContent(text="this is an error", type="text"),
            ],
            isError=True,
        ),
    ]

    @asynccontextmanager
    async def async_client_session(*args, **kwargs):
        yield client_session

    return async_client_session


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.streamablehttp_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_tool_call(
    mock_client_session_cls: AsyncMock,
    mock_streamablehttp_client: AsyncMock,
    mock_streamable_http_client_transport: Callable[
        ...,
        AsyncContextManager[Any],
    ],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests tool call."""
    # Set up the mock to return the async context manager
    mock_streamablehttp_client.side_effect = (
        mock_streamable_http_client_transport
    )
    mock_client_session_cls.side_effect = mock_client_session

    streamablehttp_provider = MCPToolProvider(
        name="mock provider",
        streamable_http_url="http://mock-url.io",
    )
    tool = MCPTool(
        provider=streamablehttp_provider,
        name="mock_tool",
        desc="mock desc",
        params_json_schema={"param1": {"type": "number"}},
    )

    # first tool call
    tool_result = await tool(
        ToolCall(tool_name=tool.name, arguments={"param1": 2}),
    )
    assert tool_result.content == [
        {"annotations": None, "text": "42", "type": "text"},
    ]
    assert tool_result.error is False

    # second tool call is an error
    tool_result = await tool(
        ToolCall(tool_name=tool.name, arguments={"param1": -2}),
    )
    assert tool_result.content == [
        {"annotations": None, "text": "this is an error", "type": "text"},
    ]
    assert tool_result.error is True
