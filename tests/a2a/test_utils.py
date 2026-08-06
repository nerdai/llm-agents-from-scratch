"""Unit tests for A2A data-conversion utilities."""

from a2a.types import Artifact, Part, Task, TaskState, TaskStatus

from llm_agents_from_scratch.a2a.utils import parts_text, task_content


def test_parts_text_joins_text_parts() -> None:
    """Tests parts_text joins text across multiple Parts."""
    parts = [Part(text="hello"), Part(text="world")]

    assert parts_text(parts) == "hello\nworld"


def test_parts_text_skips_non_text_parts() -> None:
    """Tests parts_text skips parts with no text."""
    parts = [Part(text="hello"), Part(), Part(text="world")]

    assert parts_text(parts) == "hello\nworld"


def test_parts_text_empty() -> None:
    """Tests parts_text returns an empty string for no parts."""
    assert parts_text([]) == ""


def test_task_content_concatenates_across_artifacts() -> None:
    """Tests task_content flattens text across multiple artifacts."""
    task = Task(
        id="t1",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
        artifacts=[
            Artifact(artifact_id="a1", parts=[Part(text="first")]),
            Artifact(artifact_id="a2", parts=[Part(text="second")]),
        ],
    )

    assert task_content(task) == "first\nsecond"


def test_task_content_no_artifacts() -> None:
    """Tests task_content returns an empty string for no artifacts."""
    task = Task(
        id="t1",
        status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
    )

    assert task_content(task) == ""
