from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, Callable
from unittest.mock import AsyncMock, MagicMock

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
    """Mock streamablehttp_client() async context manager."""

    @asynccontextmanager
    async def async_context_manager(*args, **kwargs):
        mock_read = AsyncMock()
        mock_write = AsyncMock()
        mock_id_callback = MagicMock()
        yield (mock_read, mock_write, mock_id_callback)

    return async_context_manager
