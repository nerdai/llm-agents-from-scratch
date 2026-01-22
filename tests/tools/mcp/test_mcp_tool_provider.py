"""Unit tests for MCPToolProvider."""

from typing import Any, AsyncContextManager
from unittest.mock import AsyncMock, patch

import pytest
from mcp import StdioServerParameters

from llm_agents_from_scratch.errors import (
    MCPWarning,
    MissingMCPServerParamsError,
)
from llm_agents_from_scratch.tools.mcp import MCPToolProvider


def test_mcp_tool_provider_init() -> None:
    """Tests initialization of an MCPToolProvider."""
    streamable_http_provider = MCPToolProvider(
        name="mock provider",
        streamable_http_url="https://mock-server-url.io",
    )

    assert streamable_http_provider.name == "mock provider"
    assert streamable_http_provider.stdio_params is None
    assert (
        streamable_http_provider.streamable_http_url
        == "https://mock-server-url.io"
    )

    stdio_params = StdioServerParameters(
        command="uv run",
        args=["fake.py"],
    )
    stdio_provider = MCPToolProvider(
        name="mock provider",
        stdio_params=stdio_params,
    )
    assert stdio_provider.name == "mock provider"
    assert stdio_provider.streamable_http_url is None
    assert stdio_provider.stdio_params == stdio_params


def test_mcp_tool_provider_init_raises_error() -> None:
    """Tests initialization raises error if no connection details provided."""
    with pytest.raises(
        MissingMCPServerParamsError,
        match="You must supply at least one",
    ):
        MCPToolProvider(name="invalid provider")


def test_mcp_tool_provider_init_emits_warning() -> None:
    """Tests init emits warning if both connection details provided."""
    with pytest.warns(
        MCPWarning,
        match="Both `stdio_params` and `streamable_http_url`",
    ):
        stdio_params = StdioServerParameters(
            command="uv run",
            args=["fake.py"],
        )
        MCPToolProvider(
            name="mock provider",
            stdio_params=stdio_params,
            streamable_http_url="https://mock-server-url.io",
        )


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_session_creation(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: AsyncContextManager[AsyncMock],
) -> None:
    """Tests creation of sessions."""
    # Set up the mock to return the async context manager
    mock_stdio_client.return_value = mock_stdio_client_transport
    mock_client_session_cls.return_value = mock_client_session

    stdio_params = StdioServerParameters(
        command="uv run",
        args=["fake.py"],
    )
    stdio_provider = MCPToolProvider(
        name="mock provider",
        stdio_params=stdio_params,
    )

    async with stdio_provider.session() as _session:
        pass

    mock_stdio_client.assert_called_once()
    mock_client_session_cls.assert_called_once()


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.streamablehttp_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_session_creation_streamable_http(
    mock_client_session_cls: AsyncMock,
    mock_streamablehttp_client: AsyncMock,
    mock_streamable_http_client_transport: AsyncContextManager[Any],
    mock_client_session: AsyncContextManager[AsyncMock],
) -> None:
    """Tests creation of sessions."""
    # Set up the mock to return the async context manager
    mock_streamablehttp_client.return_value = (
        mock_streamable_http_client_transport
    )
    mock_client_session_cls.return_value = mock_client_session

    streamablehttp_provider = MCPToolProvider(
        name="mock provider",
        streamable_http_url="http://mock-url.io",
    )

    async with streamablehttp_provider.session() as _session:
        pass

    mock_streamablehttp_client.assert_called_once()
    mock_client_session_cls.assert_called_once()


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_list_tools_stdio_client(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: AsyncContextManager[AsyncMock],
) -> None:
    """Tests creation of sessions."""
    # Set up the mock to return the async context manager
    mock_stdio_client.return_value = mock_stdio_client_transport
    mock_client_session_cls.return_value = mock_client_session

    stdio_params = StdioServerParameters(
        command="uv run",
        args=["fake.py"],
    )
    stdio_provider = MCPToolProvider(
        name="mock_provider",
        stdio_params=stdio_params,
    )

    mcp_tools = await stdio_provider.get_tools()

    assert len(mcp_tools) == 1
    assert mcp_tools[0].description == "mock_desc"
    assert mcp_tools[0].name == "mock_provider.mock_tool"
    assert mcp_tools[0].parameters_json_schema == {"param1": {"type": "number"}}
    assert mcp_tools[0].additional_annotations is None
