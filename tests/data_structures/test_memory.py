from datetime import datetime
from unittest.mock import MagicMock

from llm_agents_from_scratch.data_structures import (
    Task,
    TaskResult,
)
from llm_agents_from_scratch.data_structures.memory import Episode


def test_episode_str() -> None:
    """Tests Episode.__str__ produces prompt-ready XML with key fields."""
    task = Task(instruction="summarise the doc")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="here is the summary"),
    )

    s = str(ep)

    assert "<episode>" in s
    assert "<task>summarise the doc</task>" in s
    assert "<result>here is the summary\n    </result>" in s
    assert ep.completed_at.strftime("%Y-%m-%d") in s


def test_format_concat_completed_at() -> None:
    task = Task(instruction="task")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="result"),
        completed_at=datetime(2025, 6, 1, 12, 0, 0),
    )
    text = ep.format(mode="concat", include=["completed_at"])
    assert "2025-06-01 12:00:00" in text


def test_format_concat_rollout() -> None:
    task = Task(instruction="task")
    ep = Episode(
        task=task,
        rollout="step1 -> step2",
        result=TaskResult(task_id=task.id_, content="result"),
    )
    text = ep.format(mode="concat", include=["rollout"])
    assert "step1 -> step2" in text


def test_format_xml_metadata() -> None:
    task = Task(instruction="task")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="result"),
        metadata={"reflection": "key lesson here"},
    )
    text = ep.format(mode="xml", include=["metadata"])
    assert "<reflection>key lesson here</reflection>" in text


def test_format_xml_rollout() -> None:
    task = Task(instruction="task")
    ep = Episode(
        task=task,
        rollout="step1 -> step2",
        result=TaskResult(task_id=task.id_, content="result"),
    )
    text = ep.format(mode="xml", include=["rollout"])
    assert "<rollout>step1 -> step2</rollout>" in text


def test_format_xml_error() -> None:
    task = Task(instruction="task")
    err = RuntimeError("something went wrong")
    ep = Episode(task=task, rollout="", error=err)
    text = ep.format(mode="xml", include=["error"])
    assert "<error>something went wrong" in text


def test_format_xml_error_omitted_when_none() -> None:
    task = Task(instruction="task")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="ok"),
    )
    text = ep.format(mode="xml", include=["error"])
    assert "<error>" not in text


def test_format_concat_error() -> None:
    task = Task(instruction="task")
    err = ValueError("bad value")
    ep = Episode(task=task, rollout="", error=err)
    text = ep.format(mode="concat", include=["error"])
    assert "bad value" in text


def test_format_concat_error_omitted_when_none() -> None:
    task = Task(instruction="task")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="ok"),
    )
    text = ep.format(mode="concat", include=["error"])
    assert text == ""


def test_episode_init() -> None:
    """Tests construction of Episode."""
    mock_task = MagicMock(spec=Task)
    mock_task_result = MagicMock(spec=TaskResult)
    mock_rollout = ""
    now = datetime.now()

    episode = Episode(
        task=mock_task,
        rollout=mock_rollout,  # type: ignore
        result=mock_task_result,
        completed_at=now,
    )

    assert episode.completed_at == now
    assert episode.task == mock_task
    assert episode.result == mock_task_result
    assert episode.rollout == mock_rollout
