"""Unit tests for MCPToolProvider."""

from contextlib import asynccontextmanager
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
) -> None:
    """Tests creation of sessions."""

    # mock stdio_client() async tonctext manager
    mock_read = AsyncMock()
    mock_write = AsyncMock()

    @asynccontextmanager
    async def async_context_manager(*args, **kwargs):
        yield (mock_read, mock_write)

    # Set up the mock to return the async context manager
    mock_stdio_client.return_value = async_context_manager()

    # mock ClientSession
    client_session = AsyncMock()

    @asynccontextmanager
    async def async_client_session(*args, **kwargs):
        yield client_session

    mock_client_session_cls.return_value = async_client_session()

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
