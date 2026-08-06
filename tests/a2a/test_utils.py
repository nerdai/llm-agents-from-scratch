"""Unit tests for A2A data-conversion utilities."""

import json

from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskState,
    TaskStatus,
)

from llm_agents_from_scratch.a2a.utils import (
    a2a_response_to_tool_call_result,
    a2a_task_to_tool_call_result,
    format_a2a_parts,
    format_a2a_task,
)


def test_format_a2a_parts_joins_text_parts() -> None:
    """Tests format_a2a_parts joins text across multiple Parts."""
    parts = [Part(text="hello"), Part(text="world")]

    assert format_a2a_parts(parts) == "hello\nworld"


def test_format_a2a_parts_skips_non_text_parts() -> None:
    """Tests format_a2a_parts skips parts with no text."""
    parts = [Part(text="hello"), Part(), Part(text="world")]

    assert format_a2a_parts(parts) == "hello\nworld"


def test_format_a2a_parts_empty() -> None:
    """Tests format_a2a_parts returns an empty string for no parts."""
    assert format_a2a_parts([]) == ""


def test_format_a2a_task_concatenates_across_artifacts() -> None:
    """Tests format_a2a_task flattens text across multiple artifacts."""
    task = Task(
        id="t1",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        artifacts=[
            Artifact(artifact_id="a1", parts=[Part(text="first")]),
            Artifact(artifact_id="a2", parts=[Part(text="second")]),
        ],
    )

    assert format_a2a_task(task) == "first\nsecond"


def test_format_a2a_task_no_artifacts() -> None:
    """Tests format_a2a_task returns an empty string for no artifacts."""
    task = Task(
        id="t1",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
    )

    assert format_a2a_task(task) == ""


def test_a2a_response_to_tool_call_result_none_response_is_error() -> None:
    """Tests returns an error for a None response."""
    result = a2a_response_to_tool_call_result(None, "researcher", "tc1")

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "A2AEmptyResponseError"


def test_a2a_response_to_tool_call_result_message_payload() -> None:
    """Tests returns success for a plain Message payload."""
    response = StreamResponse(
        message=Message(role=Role.ROLE_AGENT, parts=[Part(text="hi")]),
    )

    result = a2a_response_to_tool_call_result(response, "researcher", "tc1")

    assert result.error is False
    assert result.content == "hi"


def test_a2a_response_to_tool_call_result_unset_payload_is_error() -> None:
    """Tests returns an error when no oneof field is set."""
    result = a2a_response_to_tool_call_result(
        StreamResponse(),
        "researcher",
        "tc1",
    )

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "A2AEmptyResponseError"


def test_a2a_response_to_tool_call_result_task_payload_delegates() -> None:
    """Tests delegates a Task payload to a2a_task_to_tool_call_result."""
    task = Task(
        id="t1",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        artifacts=[Artifact(artifact_id="a1", parts=[Part(text="42")])],
    )
    response = StreamResponse(task=task)

    result = a2a_response_to_tool_call_result(response, "researcher", "tc1")

    assert result.error is False
    assert result.content == "42"


def test_a2a_task_to_tool_call_result_completed() -> None:
    """Tests returns success with artifact content."""
    task = Task(
        id="t1",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        artifacts=[Artifact(artifact_id="a1", parts=[Part(text="42")])],
    )

    result = a2a_task_to_tool_call_result(task, "researcher", "tc1")

    assert result.error is False
    assert result.content == "42"


def test_a2a_task_to_tool_call_result_input_required() -> None:
    """Tests wraps the peer's question and task id."""
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

    result = a2a_task_to_tool_call_result(task, "researcher", "tc1")

    assert result.error is False
    assert "What city?" in result.content
    assert "t2" in result.content
    assert "researcher" in result.content


def test_a2a_task_to_tool_call_result_falls_back_to_task_content() -> None:
    """Tests input_required falls back to format_a2a_task when unset."""
    task = Task(
        id="t3",
        status=TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED),
        artifacts=[
            Artifact(artifact_id="a1", parts=[Part(text="Which city?")]),
        ],
    )

    result = a2a_task_to_tool_call_result(task, "researcher", "tc1")

    assert "Which city?" in result.content


def test_a2a_task_to_tool_call_result_failed_with_message() -> None:
    """Tests maps FAILED to the shared error JSON shape."""
    task = Task(
        id="t4",
        status=TaskStatus(
            state=TaskState.TASK_STATE_FAILED,
            message=Message(
                role=Role.ROLE_AGENT,
                parts=[Part(text="Something went wrong.")],
            ),
        ),
    )

    result = a2a_task_to_tool_call_result(task, "researcher", "tc1")

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "TASK_STATE_FAILED"
    assert details["a2a_agent"] == "researcher"
    assert details["message"] == "Something went wrong."


def test_a2a_task_to_tool_call_result_failed_generic_text() -> None:
    """Tests falls back to a generic message when unset."""
    task = Task(id="t5", status=TaskStatus(state=TaskState.TASK_STATE_FAILED))

    result = a2a_task_to_tool_call_result(task, "researcher", "tc1")

    assert result.error is True
    details = json.loads(result.content)
    assert "TASK_STATE_FAILED" in details["message"]


def test_a2a_task_to_tool_call_result_unrecognized_state() -> None:
    """Tests an unrecognized TaskState.state value is handled, not raised."""
    task = Task(id="t6", status=TaskStatus(state=999))

    result = a2a_task_to_tool_call_result(task, "researcher", "tc1")

    assert result.error is True
    details = json.loads(result.content)
    assert "999" in details["error_type"]
