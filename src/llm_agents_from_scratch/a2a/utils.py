"""A2A data-conversion utilities."""

import json
from collections.abc import Iterable

from a2a.types import Artifact, Part, StreamResponse, TaskState
from a2a.types import Task as A2ATask

from llm_agents_from_scratch.data_structures import ToolCallResult

from .constants import A2A_INPUT_REQUIRED_TEMPLATE


def a2a_parts_text(parts: Iterable[Part]) -> str:
    """Join the text of every text Part, skipping non-text parts.

    Args:
        parts (Iterable[Part]): Parts to extract text from.

    Returns:
        str: The joined text, newline-separated.
    """
    return "\n".join(p.text for p in parts if p.text)


def a2a_artifacts_text(artifacts: Iterable[Artifact]) -> str:
    """Join the text of every Part across a collection of Artifacts.

    Args:
        artifacts (Iterable[Artifact]): Artifacts to extract text from.

    Returns:
        str: The joined text across all artifacts, newline-separated.
    """
    parts = [p for artifact in artifacts for p in artifact.parts]
    return a2a_parts_text(parts)


def a2a_response_to_tool_call_result(
    response: StreamResponse | None,
    agent_name: str,
    tool_call_id: str,
) -> ToolCallResult:
    """Turn the final ``StreamResponse`` chunk into a ``ToolCallResult``.

    Args:
        response (StreamResponse | None): The last chunk yielded by
            ``client.send_message()``, or ``None`` if the peer yielded
            nothing.
        agent_name (str): The dispatched-to peer's registry name, for
            error JSON and the input-required template.
        tool_call_id (str): The originating tool call's id.

    Returns:
        ToolCallResult: Success with the peer's content, an
            ``A2A_INPUT_REQUIRED_TEMPLATE``-wrapped result, or an error
            result.
    """
    if response is None:
        return ToolCallResult(
            tool_call_id=tool_call_id,
            error=True,
            content=json.dumps(
                {
                    "error_type": "A2AEmptyResponseError",
                    "a2a_agent": agent_name,
                    "message": "Peer returned no task or message.",
                },
            ),
        )

    kind = response.WhichOneof("payload")

    if kind == "message":
        return ToolCallResult(
            tool_call_id=tool_call_id,
            error=False,
            content=a2a_parts_text(response.message.parts),
        )

    if kind == "task":
        return a2a_task_to_tool_call_result(
            response.task,
            agent_name,
            tool_call_id,
        )

    return ToolCallResult(
        tool_call_id=tool_call_id,
        error=True,
        content=json.dumps(
            {
                "error_type": "A2AEmptyResponseError",
                "a2a_agent": agent_name,
                "message": "Peer returned no task or message.",
            },
        ),
    )


def a2a_task_to_tool_call_result(
    task: A2ATask,
    agent_name: str,
    tool_call_id: str,
) -> ToolCallResult:
    """Turn a peer's final ``A2ATask`` into a ``ToolCallResult``.

    Args:
        task (A2ATask): The peer's final ``A2ATask`` state.
        agent_name (str): The dispatched-to peer's registry name, for
            error JSON and the input-required template.
        tool_call_id (str): The originating tool call's id.

    Returns:
        ToolCallResult: Success with the task's artifact content, an
            ``A2A_INPUT_REQUIRED_TEMPLATE``-wrapped result, or an error
            result for any other terminal state.
    """
    try:
        state = TaskState.Name(task.status.state)
    except ValueError:
        # A peer running a newer SDK/protocol version than ours could
        # report a state we don't recognize. Fall through to the
        # generic error branch below rather than raising.
        state = f"TASK_STATE_UNKNOWN_{task.status.state}"

    if state == "TASK_STATE_COMPLETED":
        return ToolCallResult(
            tool_call_id=tool_call_id,
            error=False,
            content=a2a_artifacts_text(task.artifacts),
        )

    if state == "TASK_STATE_INPUT_REQUIRED":
        question = a2a_parts_text(
            task.status.message.parts,
        ) or a2a_artifacts_text(
            task.artifacts,
        )
        return ToolCallResult(
            tool_call_id=tool_call_id,
            error=False,
            content=A2A_INPUT_REQUIRED_TEMPLATE.format(
                name=agent_name,
                question=question,
                task_id=task.id,
            ),
        )

    # any other state (FAILED, CANCELED, REJECTED, AUTH_REQUIRED, or an
    # unexpected non-terminal state from a non-streaming response) is an
    # error the coordinator should re-plan around
    message_text = (
        a2a_parts_text(
            task.status.message.parts,
        )
        or f"Task ended in state {state}."
    )
    return ToolCallResult(
        tool_call_id=tool_call_id,
        error=True,
        content=json.dumps(
            {
                "error_type": state,
                "a2a_agent": agent_name,
                "message": message_text,
            },
        ),
    )
