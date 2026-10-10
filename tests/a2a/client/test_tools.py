"""Unit tests for UseA2AAgentTool."""

import json
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from a2a.types import (
    AgentCard,
    AgentInterface,
    Artifact,
    Message,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
    Task,
    TaskState,
    TaskStatus,
)

from llm_agents_from_scratch.a2a import A2AAgentSpec, UseA2AAgentTool
from llm_agents_from_scratch.data_structures import ToolCall

TOOL_NAME = "from_scratch__use_a2a_agent"


def _spec(name: str = "researcher") -> A2AAgentSpec:
    card = AgentCard(
        name=name,
        description="A peer agent.",
        supported_interfaces=[AgentInterface(url="http://peer:9999")],
    )
    return A2AAgentSpec.from_agent_card(agent_card=card)


def _fake_client(*responses: StreamResponse) -> MagicMock:
    async def send_message(
        request: SendMessageRequest,
    ) -> AsyncIterator[StreamResponse]:
        for response in responses:
            yield response

    client = MagicMock()
    client.send_message = send_message
    client.close = AsyncMock()
    return client


def _patch_create_client(client: MagicMock) -> Any:
    return patch(
        "llm_agents_from_scratch.a2a.client.tools.create_client",
        new=AsyncMock(return_value=client),
    )


def test_use_a2a_agent_tool_name() -> None:
    """Tests UseA2AAgentTool.name."""
    tool = UseA2AAgentTool(a2a_agents_registry={})
    assert tool.name == TOOL_NAME


def test_use_a2a_agent_tool_description() -> None:
    """Tests UseA2AAgentTool.description mentions dispatch."""
    tool = UseA2AAgentTool(a2a_agents_registry={})
    assert "dispatch" in tool.description.lower()


def test_use_a2a_agent_tool_schema_enum() -> None:
    """Tests parameters_json_schema enum contains registered peer names."""
    registry = {"researcher": _spec("researcher"), "coder": _spec("coder")}
    tool = UseA2AAgentTool(a2a_agents_registry=registry)
    enum = tool.parameters_json_schema["properties"]["name"]["enum"]
    assert "researcher" in enum
    assert "coder" in enum


def test_use_a2a_agent_tool_schema_required() -> None:
    """Tests schema requires name and task; task_id stays optional."""
    tool = UseA2AAgentTool(a2a_agents_registry={})
    schema = tool.parameters_json_schema
    assert schema["required"] == ["name", "task"]
    assert "task_id" in schema["properties"]


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_dispatches_task_completed() -> None:
    """Tests a COMPLETED task response returns its artifact text."""
    task = Task(
        id="t1",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        artifacts=[Artifact(artifact_id="a1", parts=[Part(text="42")])],
    )
    client = _fake_client(StreamResponse(task=task))

    with _patch_create_client(client):
        tool = UseA2AAgentTool(a2a_agents_registry={"researcher": _spec()})
        tool_call = ToolCall(
            tool_name=TOOL_NAME,
            arguments={"name": "researcher", "task": "What is the answer?"},
        )
        result = await tool(tool_call=tool_call)

    assert result.error is False
    assert result.content == "42"
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_dispatches_immediate_message() -> None:
    """Tests a plain Message response (no Task) returns its text."""
    message = Message(role=Role.ROLE_AGENT, parts=[Part(text="quick reply")])
    client = _fake_client(StreamResponse(message=message))

    with _patch_create_client(client):
        tool = UseA2AAgentTool(a2a_agents_registry={"researcher": _spec()})
        tool_call = ToolCall(
            tool_name=TOOL_NAME,
            arguments={"name": "researcher", "task": "hello"},
        )
        result = await tool(tool_call=tool_call)

    assert result.error is False
    assert result.content == "quick reply"


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_input_required_returns_template() -> None:
    """Tests INPUT_REQUIRED returns a template echoing the task id."""
    task = Task(
        id="t2",
        status=TaskStatus(
            state=TaskState.TASK_STATE_INPUT_REQUIRED,
            message=Message(
                role=Role.ROLE_AGENT,
                parts=[Part(text="What city?")],
            ),
        ),
    )
    client = _fake_client(StreamResponse(task=task))

    with _patch_create_client(client):
        tool = UseA2AAgentTool(a2a_agents_registry={"researcher": _spec()})
        tool_call = ToolCall(
            tool_name=TOOL_NAME,
            arguments={"name": "researcher", "task": "Book a flight."},
        )
        result = await tool(tool_call=tool_call)

    assert result.error is False
    assert "What city?" in result.content
    assert "t2" in result.content
    assert "researcher" in result.content


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_failed_state_returns_error() -> None:
    """Tests a FAILED task response is returned as an error result."""
    task = Task(
        id="t3",
        status=TaskStatus(
            state=TaskState.TASK_STATE_FAILED,
            message=Message(
                role=Role.ROLE_AGENT,
                parts=[Part(text="Something went wrong.")],
            ),
        ),
    )
    client = _fake_client(StreamResponse(task=task))

    with _patch_create_client(client):
        tool = UseA2AAgentTool(a2a_agents_registry={"researcher": _spec()})
        tool_call = ToolCall(
            tool_name=TOOL_NAME,
            arguments={"name": "researcher", "task": "do it"},
        )
        result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "TASK_STATE_FAILED"
    assert details["a2a_agent"] == "researcher"
    assert details["message"] == "Something went wrong."


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_empty_response_is_error() -> None:
    """Tests a StreamResponse with neither task nor message is an error."""
    client = _fake_client(StreamResponse())

    with _patch_create_client(client):
        tool = UseA2AAgentTool(a2a_agents_registry={"researcher": _spec()})
        tool_call = ToolCall(
            tool_name=TOOL_NAME,
            arguments={"name": "researcher", "task": "do it"},
        )
        result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "A2AEmptyResponseError"


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_unknown_name() -> None:
    """Tests unknown peer name returns error result.

    Caught by the ``name`` enum in ``parameters_json_schema``, which
    lists the valid peer names directly in the ``ValidationError``
    message.
    """
    tool = UseA2AAgentTool(a2a_agents_registry={"researcher": _spec()})
    tool_call = ToolCall(
        tool_name=TOOL_NAME,
        arguments={"name": "unknown", "task": "do it"},
    )
    result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "ValidationError"
    assert "unknown" in details["message"]


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_missing_name_arg() -> None:
    """Tests missing name argument returns error result."""
    tool = UseA2AAgentTool(a2a_agents_registry={})
    tool_call = ToolCall(tool_name=TOOL_NAME, arguments={"task": "do it"})
    result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert "'name'" in details["message"]


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_missing_task_arg() -> None:
    """Tests missing task argument returns error result."""
    tool = UseA2AAgentTool(a2a_agents_registry={})
    tool_call = ToolCall(
        tool_name=TOOL_NAME,
        arguments={"name": "researcher"},
    )
    result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert "'task'" in details["message"]


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_invalid_task_id_type() -> None:
    """Tests a non-string task_id returns error result.

    jsonschema's default type-violation message doesn't name the
    property (unlike its ``required`` messages), so this checks the
    error shape rather than a ``'task_id'`` substring.
    """
    tool = UseA2AAgentTool(a2a_agents_registry={"researcher": _spec()})
    tool_call = ToolCall(
        tool_name=TOOL_NAME,
        arguments={"name": "researcher", "task": "do it", "task_id": 123},
    )
    result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "ValidationError"
    assert "is not of type 'string'" in details["message"]


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_passes_task_id_to_resume() -> None:
    """Tests a supplied task_id is forwarded on the outgoing message."""
    captured: dict[str, object] = {}

    async def send_message(
        request: SendMessageRequest,
    ) -> AsyncIterator[StreamResponse]:
        captured["task_id"] = request.message.task_id
        task = Task(
            id=request.message.task_id,
            status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            artifacts=[Artifact(artifact_id="a", parts=[Part(text="ok")])],
        )
        yield StreamResponse(task=task)

    client = MagicMock()
    client.send_message = send_message
    client.close = AsyncMock()

    with _patch_create_client(client):
        tool = UseA2AAgentTool(a2a_agents_registry={"researcher": _spec()})
        tool_call = ToolCall(
            tool_name=TOOL_NAME,
            arguments={
                "name": "researcher",
                "task": "Paris",
                "task_id": "t2",
            },
        )
        result = await tool(tool_call=tool_call)

    assert captured["task_id"] == "t2"
    assert result.error is False


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_catches_unexpected_error() -> None:
    """Tests an exception raised during dispatch is caught as an error."""
    with patch(
        "llm_agents_from_scratch.a2a.client.tools.create_client",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        tool = UseA2AAgentTool(a2a_agents_registry={"researcher": _spec()})
        tool_call = ToolCall(
            tool_name=TOOL_NAME,
            arguments={"name": "researcher", "task": "do it"},
        )
        result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "RuntimeError"
    assert details["a2a_agent"] == "researcher"
    assert "boom" in details["message"]


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_closes_client_on_error() -> None:
    """Tests the client is still closed when the peer returns a failure."""
    task = Task(id="t4", status=TaskStatus(state=TaskState.TASK_STATE_FAILED))
    client = _fake_client(StreamResponse(task=task))

    with _patch_create_client(client):
        tool = UseA2AAgentTool(a2a_agents_registry={"researcher": _spec()})
        tool_call = ToolCall(
            tool_name=TOOL_NAME,
            arguments={"name": "researcher", "task": "do it"},
        )
        await tool(tool_call=tool_call)

    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_closes_httpx_client_on_create_failure() -> (
    None
):
    """Tests the raw httpx client is closed if create_client() itself fails."""
    created_httpx_clients: list[MagicMock] = []

    class _TrackedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created_httpx_clients.append(self)

    with (
        patch(
            "llm_agents_from_scratch.a2a.client.tools.httpx.AsyncClient",
            new=_TrackedAsyncClient,
        ),
        patch(
            "llm_agents_from_scratch.a2a.client.tools.create_client",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ),
    ):
        tool = UseA2AAgentTool(a2a_agents_registry={"researcher": _spec()})
        tool_call = ToolCall(
            tool_name=TOOL_NAME,
            arguments={"name": "researcher", "task": "do it"},
        )
        result = await tool(tool_call=tool_call)

    assert result.error is True
    assert len(created_httpx_clients) == 1
    assert created_httpx_clients[0].is_closed


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_passes_spec_timeout() -> None:
    """Tests spec.timeout reaches the dispatch httpx.AsyncClient."""
    captured_kwargs: dict[str, Any] = {}

    class _TrackedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
            super().__init__(*args, **kwargs)

    client = _fake_client(StreamResponse(message=Message(role=Role.ROLE_AGENT)))
    spec = A2AAgentSpec.from_agent_card(
        agent_card=AgentCard(
            name="researcher",
            description="A peer agent.",
            supported_interfaces=[AgentInterface(url="http://peer:9999")],
        ),
        timeout=123.0,
    )

    with (
        patch(
            "llm_agents_from_scratch.a2a.client.tools.httpx.AsyncClient",
            new=_TrackedAsyncClient,
        ),
        _patch_create_client(client),
    ):
        tool = UseA2AAgentTool(a2a_agents_registry={"researcher": spec})
        tool_call = ToolCall(
            tool_name=TOOL_NAME,
            arguments={"name": "researcher", "task": "do it"},
        )
        await tool(tool_call=tool_call)

    assert captured_kwargs["timeout"] == 123.0  # noqa: PLR2004


@pytest.mark.asyncio
async def test_use_a2a_agent_tool_suppresses_close_error() -> None:
    """Tests a close()-time exception doesn't override a computed result."""
    task = Task(
        id="t5",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        artifacts=[Artifact(artifact_id="a1", parts=[Part(text="42")])],
    )
    client = _fake_client(StreamResponse(task=task))
    client.close = AsyncMock(side_effect=RuntimeError("close boom"))

    with _patch_create_client(client):
        tool = UseA2AAgentTool(a2a_agents_registry={"researcher": _spec()})
        tool_call = ToolCall(
            tool_name=TOOL_NAME,
            arguments={"name": "researcher", "task": "do it"},
        )
        result = await tool(tool_call=tool_call)

    assert result.error is False
    assert result.content == "42"
