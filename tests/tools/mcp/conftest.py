from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import ClientSession, ListToolsResult, Tool


@pytest.fixture()
def mock_stdio_client_transport() -> AsyncContextManager[Any]:
    """Mock stdio_client() async context manager."""
    mock_read = AsyncMock()
    mock_write = AsyncMock()

    @asynccontextmanager
    async def async_context_manager(*args, **kwargs):
        yield (mock_read, mock_write)

    return async_context_manager()


@pytest.fixture()
def mock_streamable_http_client_transport() -> AsyncContextManager[Any]:
    """Mock streamablehttp_client() async context manager."""
    mock_read = AsyncMock()
    mock_write = AsyncMock()
    mock_id_callback = MagicMock()

    @asynccontextmanager
    async def async_context_manager(*args, **kwargs):
        yield (mock_read, mock_write, mock_id_callback)

    return async_context_manager()


@pytest.fixture()
def mock_client_session() -> AsyncContextManager[AsyncMock]:
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
    client_session.list_tools = mock_list_tools

    @asynccontextmanager
    async def async_client_session(*args, **kwargs):
        yield client_session

    return async_client_session()
