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

from llm_agents_from_scratch.a2a.client.utils import (
    a2a_artifacts_text,
    a2a_parts_text,
    a2a_response_to_tool_call_result,
    a2a_task_to_tool_call_result,
)


def test_a2a_parts_text_joins_text_parts() -> None:
    """Tests a2a_parts_text joins text across multiple Parts."""
    parts = [Part(text="hello"), Part(text="world")]

    assert a2a_parts_text(parts) == "hello\nworld"


def test_a2a_parts_text_skips_non_text_parts() -> None:
    """Tests a2a_parts_text skips parts with no text."""
    parts = [Part(text="hello"), Part(), Part(text="world")]

    assert a2a_parts_text(parts) == "hello\nworld"


def test_a2a_parts_text_empty() -> None:
    """Tests a2a_parts_text returns an empty string for no parts."""
    assert a2a_parts_text([]) == ""


def test_a2a_artifacts_text_concatenates_across_artifacts() -> None:
    """Tests a2a_artifacts_text flattens text across multiple artifacts."""
    artifacts = [
        Artifact(artifact_id="a1", parts=[Part(text="first")]),
        Artifact(artifact_id="a2", parts=[Part(text="second")]),
    ]

    assert a2a_artifacts_text(artifacts) == "first\nsecond"


def test_a2a_artifacts_text_no_artifacts() -> None:
    """Tests a2a_artifacts_text returns an empty string for no artifacts."""
    assert a2a_artifacts_text([]) == ""


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


def test_a2a_response_to_tool_call_result_never_raises() -> None:
    """Tests an exception while parsing is caught, not propagated.

    A malformed/unexpected response (here: not even a real
    ``StreamResponse``, lacking ``WhichOneof``) is exactly the kind of
    thing a non-conformant peer could send. This should degrade to an
    error result rather than raise past this function.
    """
    result = a2a_response_to_tool_call_result(object(), "researcher", "tc1")  # type: ignore[arg-type]

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "AttributeError"
    assert details["a2a_agent"] == "researcher"


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


def test_a2a_task_to_tool_call_result_completed_falls_back() -> None:
    """Tests completed falls back to status message when no artifacts."""
    task = Task(
        id="t1",
        status=TaskStatus(
            state=TaskState.TASK_STATE_COMPLETED,
            message=Message(
                role=Role.ROLE_AGENT,
                parts=[Part(text="Done, no artifact needed.")],
            ),
        ),
    )

    result = a2a_task_to_tool_call_result(task, "researcher", "tc1")

    assert result.error is False
    assert result.content == "Done, no artifact needed."


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


def test_a2a_task_to_tool_call_result_falls_back_to_artifacts_text() -> None:
    """Tests input_required falls back to a2a_artifacts_text when unset."""
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
