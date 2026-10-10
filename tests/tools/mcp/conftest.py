from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, Callable
from unittest.mock import AsyncMock

import pytest


@pytest.fixture()
def mock_stdio_client_transport() -> Callable[..., AsyncContextManager[Any]]:
    """Mock stdio_client() async context manager."""

    @asynccontextmanager
    async def async_context_manager(*args, **kwargs):
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        yield (mock_read, mock_write)

    return async_context_manager


@pytest.fixture()
def mock_streamable_http_client_transport() -> Callable[
    ...,
    AsyncContextManager[Any],
]:
    """Mock streamable_http_client() async context manager.

    Mirrors mcp 2.0's real signature, so a call in the old 1.x shape
    (positional headers) fails here as it does at runtime, and yields the
    real 2-tuple of streams.
    """

    @asynccontextmanager
    async def async_context_manager(
        url: str,
        *,
        http_client: Any = None,
        terminate_on_close: bool = True,
    ):
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        yield (mock_read, mock_write)

    return async_context_manager
