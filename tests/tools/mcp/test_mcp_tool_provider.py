"""Unit tests for MCPToolProvider."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncContextManager, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import ClientSession, ListToolsResult, StdioServerParameters, Tool

from llm_agents_from_scratch import LLMAgentBuilder
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


@pytest.fixture()
def mock_client_session() -> Callable[..., AsyncContextManager[AsyncMock]]:
    """Mock ClientSession."""

    @asynccontextmanager
    async def async_client_session(*args, **kwargs):
        client_session = AsyncMock(spec=ClientSession)
        mock_list_tools = AsyncMock()
        mock_list_tools.return_value = ListToolsResult(
            tools=[
                Tool(
                    name="mock_tool",
                    description="mock_desc",
                    input_schema={"param1": {"type": "number"}},
                ),
            ],
        )
        client_session.list_tools = mock_list_tools
        yield client_session

    return async_client_session


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_session_creation(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests creation of sessions."""
    # Set up the mock to return the async context manager
    mock_stdio_client.side_effect = mock_stdio_client_transport
    mock_client_session_cls.side_effect = mock_client_session

    stdio_params = StdioServerParameters(
        command="uv run",
        args=["fake.py"],
    )
    stdio_provider = MCPToolProvider(
        name="mock provider",
        stdio_params=stdio_params,
    )

    await stdio_provider.session()

    mock_stdio_client.assert_called_once()
    mock_client_session_cls.assert_called_once()
    assert stdio_provider._session_ready.is_set()


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.streamable_http_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_session_creation_streamable_http(
    mock_client_session_cls: AsyncMock,
    mock_streamable_http_client: AsyncMock,
    mock_streamable_http_client_transport: Callable[
        ...,
        AsyncContextManager[Any],
    ],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests creation of sessions."""
    # Set up the mock to return the async context manager
    mock_streamable_http_client.side_effect = (
        mock_streamable_http_client_transport
    )
    mock_client_session_cls.side_effect = mock_client_session

    streamablehttp_provider = MCPToolProvider(
        name="mock provider",
        streamable_http_url="http://mock-url.io",
    )

    await streamablehttp_provider.session()

    mock_streamable_http_client.assert_called_once()
    mock_client_session_cls.assert_called_once()
    assert streamablehttp_provider._session_ready.is_set()


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_session_creation_raises_error(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests creation of sessions."""
    # Set up the mock to return the async context manager
    mock_stdio_client.side_effect = mock_stdio_client_transport
    mock_client_session_cls.side_effect = FileNotFoundError()

    stdio_params = StdioServerParameters(
        command="uv run",
        args=["fake.py"],
    )
    stdio_provider = MCPToolProvider(
        name="mock provider",
        stdio_params=stdio_params,
    )

    with pytest.raises(FileNotFoundError):
        await stdio_provider.session()


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_list_tools_stdio_client(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests creation of sessions."""
    # Set up the mock to return the async context manager
    mock_stdio_client.side_effect = mock_stdio_client_transport
    mock_client_session_cls.side_effect = mock_client_session

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
    assert mcp_tools[0].name == "mcp__mock_provider__mock_tool"
    assert mcp_tools[0].parameters_json_schema == {"param1": {"type": "number"}}
    assert mcp_tools[0].additional_annotations is None


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_close(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Test closing of session."""
    # Set up the mock to return the async context manager
    mock_stdio_client.side_effect = mock_stdio_client_transport
    mock_client_session_cls.side_effect = mock_client_session

    stdio_params = StdioServerParameters(
        command="uv run",
        args=["fake.py"],
    )
    stdio_provider = MCPToolProvider(
        name="mock_provider",
        stdio_params=stdio_params,
    )

    await stdio_provider.session()
    assert stdio_provider._session_ready.is_set()
    assert not stdio_provider._shutdown_event.is_set()

    # act
    await stdio_provider.close()
    assert not stdio_provider._shutdown_event.is_set()
    assert stdio_provider._session is None
    assert stdio_provider._session_task is None


def _stdio_provider() -> MCPToolProvider:
    return MCPToolProvider(
        name="mock_provider",
        stdio_params=StdioServerParameters(command="uv run", args=["fake.py"]),
    )


def _live_session_tasks() -> list[asyncio.Task]:
    return [
        t
        for t in asyncio.all_tasks()
        if not t.done()
        and getattr(t.get_coro(), "__qualname__", "")
        == "MCPToolProvider._create_session"
    ]


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_session_reused_on_subsequent_calls(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests sequential calls reuse the session created by the first call."""
    mock_stdio_client.side_effect = mock_stdio_client_transport
    mock_client_session_cls.side_effect = mock_client_session
    stdio_provider = _stdio_provider()

    first = await stdio_provider.session()
    first_task = stdio_provider._session_task
    second = await stdio_provider.session()

    assert first is second
    assert stdio_provider._session_task is first_task
    mock_stdio_client.assert_called_once()
    mock_client_session_cls.assert_called_once()

    await stdio_provider.close()


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_concurrent_session_calls_share_one_session(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests concurrent first-time callers share a single session."""
    mock_stdio_client.side_effect = mock_stdio_client_transport
    mock_client_session_cls.side_effect = mock_client_session
    stdio_provider = _stdio_provider()

    sessions = await asyncio.gather(
        *(stdio_provider.session() for _ in range(5)),
    )

    assert all(s is sessions[0] for s in sessions)
    assert stdio_provider._session is sessions[0]
    mock_stdio_client.assert_called_once()
    mock_client_session_cls.assert_called_once()
    assert len(_live_session_tasks()) == 1

    await stdio_provider.close()
    assert _live_session_tasks() == []


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.streamable_http_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_concurrent_session_calls_share_one_session_streamable_http(
    mock_client_session_cls: AsyncMock,
    mock_streamable_http_client: AsyncMock,
    mock_streamable_http_client_transport: Callable[
        ...,
        AsyncContextManager[Any],
    ],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests concurrent first-time callers share one streamable HTTP session."""
    mock_streamable_http_client.side_effect = (
        mock_streamable_http_client_transport
    )
    mock_client_session_cls.side_effect = mock_client_session
    streamablehttp_provider = MCPToolProvider(
        name="mock provider",
        streamable_http_url="http://mock-url.io",
    )

    sessions = await asyncio.gather(
        *(streamablehttp_provider.session() for _ in range(5)),
    )

    assert all(s is sessions[0] for s in sessions)
    mock_streamable_http_client.assert_called_once()
    mock_client_session_cls.assert_called_once()

    await streamablehttp_provider.close()
    assert _live_session_tasks() == []


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_concurrent_session_calls_share_creation_error(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
) -> None:
    """Tests a failed creation surfaces the same error to every waiter."""
    mock_stdio_client.side_effect = mock_stdio_client_transport
    error = FileNotFoundError("mock server not found")
    mock_client_session_cls.side_effect = error
    stdio_provider = _stdio_provider()

    results = await asyncio.gather(
        *(stdio_provider.session() for _ in range(3)),
        return_exceptions=True,
    )

    assert all(r is error for r in results)
    mock_stdio_client.assert_called_once()
    mock_client_session_cls.assert_called_once()
    assert not stdio_provider._session_ready.is_set()


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_session_retries_after_failed_creation(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests a call after a failed creation starts a fresh attempt."""
    mock_stdio_client.side_effect = mock_stdio_client_transport
    mock_client_session_cls.side_effect = FileNotFoundError()
    stdio_provider = _stdio_provider()

    with pytest.raises(FileNotFoundError):
        await stdio_provider.session()
    failed_task = stdio_provider._session_task

    # act
    mock_client_session_cls.side_effect = mock_client_session
    session = await stdio_provider.session()

    assert session is not None
    assert stdio_provider._session_task is not failed_task
    assert stdio_provider._session_ready.is_set()
    assert mock_client_session_cls.call_count == 2  # noqa: PLR2004

    await stdio_provider.close()


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_session_recreated_after_close(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests session() after close() opens a new session."""
    mock_stdio_client.side_effect = mock_stdio_client_transport
    mock_client_session_cls.side_effect = mock_client_session
    stdio_provider = _stdio_provider()

    first = await stdio_provider.session()
    await stdio_provider.close()

    # act
    sessions = await asyncio.gather(
        stdio_provider.session(),
        stdio_provider.session(),
    )

    assert sessions[0] is sessions[1]
    assert sessions[0] is not first
    assert stdio_provider._session_ready.is_set()
    assert mock_stdio_client.call_count == 2  # noqa: PLR2004
    assert mock_client_session_cls.call_count == 2  # noqa: PLR2004

    await stdio_provider.close()
    assert _live_session_tasks() == []


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_concurrent_get_tools_share_one_session(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests concurrent get_tools() on an uninitialized provider."""
    mock_stdio_client.side_effect = mock_stdio_client_transport
    mock_client_session_cls.side_effect = mock_client_session
    stdio_provider = _stdio_provider()

    tool_lists = await asyncio.gather(
        stdio_provider.get_tools(),
        stdio_provider.get_tools(),
    )

    assert [len(tools) for tools in tool_lists] == [1, 1]
    mock_stdio_client.assert_called_once()
    mock_client_session_cls.assert_called_once()

    await stdio_provider.close()
    assert _live_session_tasks() == []


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_concurrent_builds_share_one_session(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests concurrent builds with an uninitialized provider share a session.

    Mirrors concurrent dispatches to a subagent whose builder is the only
    owner of the provider, since each dispatch calls ``build()``.
    """
    mock_stdio_client.side_effect = mock_stdio_client_transport
    mock_client_session_cls.side_effect = mock_client_session
    stdio_provider = _stdio_provider()
    builder = LLMAgentBuilder(llm=MagicMock(), mcp_providers=[stdio_provider])

    agents = await asyncio.gather(*(builder.build() for _ in range(3)))

    assert all(len(agent.tools) == 1 for agent in agents)
    mock_stdio_client.assert_called_once()
    mock_client_session_cls.assert_called_once()

    await stdio_provider.close()
    assert _live_session_tasks() == []


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.mcp.provider.stdio_client")
@patch("llm_agents_from_scratch.tools.mcp.provider.ClientSession")
async def test_waiter_sees_own_failure_when_another_waiter_retries(
    mock_client_session_cls: AsyncMock,
    mock_stdio_client: AsyncMock,
    mock_stdio_client_transport: AsyncContextManager[Any],
    mock_client_session: Callable[..., AsyncContextManager[AsyncMock]],
) -> None:
    """Tests a waiter still raises when a retry replaces the failed task.

    The first waiter to wake retries before the second waiter resumes, so
    ``_session_task`` already points at the new attempt by then.
    """
    mock_stdio_client.side_effect = mock_stdio_client_transport
    error = FileNotFoundError("mock server not found")
    mock_client_session_cls.side_effect = [error, mock_client_session()]
    stdio_provider = _stdio_provider()

    async def retry_on_failure() -> ClientSession:
        try:
            return await stdio_provider.session()
        except FileNotFoundError:
            return await stdio_provider.session()

    retried, second_waiter = await asyncio.gather(
        retry_on_failure(),
        stdio_provider.session(),
        return_exceptions=True,
    )

    assert second_waiter is error
    assert retried is stdio_provider._session
    assert retried is not None
    assert mock_client_session_cls.call_count == 2  # noqa: PLR2004

    await stdio_provider.close()
    assert _live_session_tasks() == []
