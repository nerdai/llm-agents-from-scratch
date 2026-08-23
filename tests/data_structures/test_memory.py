from datetime import datetime
from unittest.mock import MagicMock

from llm_agents_from_scratch.data_structures import (
    Task,
    TaskResult,
)
from llm_agents_from_scratch.data_structures.memory import (
    Episode,
    EpisodeFormatMode,
)

_ALL_FIELDS = set(Episode.model_fields)


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
    assert "<result>here is the summary</result>" in s
    assert ep.completed_at.strftime("%Y-%m-%d") in s


def test_format_concat_labels() -> None:
    task = Task(instruction="my task")
    ep = Episode(
        task=task,
        rollout="step1",
        result=TaskResult(task_id=task.id_, content="my result"),
        metadata={"reflection": "a lesson"},
        completed_at=datetime(2025, 6, 1, 12, 0, 0),
    )
    text = ep.format(mode=EpisodeFormatMode.CONCAT, exclude={"id_"})
    assert "task: my task" in text
    assert "result: my result" in text
    assert "reflection: a lesson" in text
    assert "completed_at: 2025-06-01 12:00:00" in text
    assert "rollout: step1" in text


def test_format_concat_completed_at() -> None:
    task = Task(instruction="task")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="result"),
        completed_at=datetime(2025, 6, 1, 12, 0, 0),
    )
    text = ep.format(
        mode=EpisodeFormatMode.CONCAT,
        exclude=_ALL_FIELDS - {"completed_at"},
    )
    assert "2025-06-01 12:00:00" in text


def test_format_concat_rollout() -> None:
    task = Task(instruction="task")
    ep = Episode(
        task=task,
        rollout="step1 -> step2",
        result=TaskResult(task_id=task.id_, content="result"),
    )
    text = ep.format(
        mode=EpisodeFormatMode.CONCAT,
        exclude=_ALL_FIELDS - {"rollout"},
    )
    assert "step1 -> step2" in text


def test_format_xml_metadata() -> None:
    task = Task(instruction="task")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="result"),
        metadata={"reflection": "key lesson here"},
    )
    text = ep.format(
        mode=EpisodeFormatMode.XML,
        exclude=_ALL_FIELDS - {"metadata"},
    )
    assert "<reflection>key lesson here</reflection>" in text


def test_format_xml_rollout() -> None:
    task = Task(instruction="task")
    ep = Episode(
        task=task,
        rollout="step1 -> step2",
        result=TaskResult(task_id=task.id_, content="result"),
    )
    text = ep.format(
        mode=EpisodeFormatMode.XML,
        exclude=_ALL_FIELDS - {"rollout"},
    )
    assert "<rollout>step1 -> step2</rollout>" in text


def test_format_xml_error() -> None:
    task = Task(instruction="task")
    err = RuntimeError("something went wrong")
    ep = Episode(task=task, rollout="", error=err)
    text = ep.format(
        mode=EpisodeFormatMode.XML,
        exclude=_ALL_FIELDS - {"error"},
    )
    assert "<error>something went wrong" in text


def test_format_xml_error_omitted_when_none() -> None:
    task = Task(instruction="task")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="ok"),
    )
    text = ep.format(
        mode=EpisodeFormatMode.XML,
        exclude=_ALL_FIELDS - {"error"},
    )
    assert "<error>" not in text


def test_format_concat_error() -> None:
    task = Task(instruction="task")
    err = ValueError("bad value")
    ep = Episode(task=task, rollout="", error=err)
    text = ep.format(
        mode=EpisodeFormatMode.CONCAT,
        exclude=_ALL_FIELDS - {"error"},
    )
    assert "bad value" in text


def test_format_concat_error_omitted_when_none() -> None:
    task = Task(instruction="task")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="ok"),
    )
    text = ep.format(
        mode=EpisodeFormatMode.CONCAT,
        exclude=_ALL_FIELDS - {"error"},
    )
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


def test_episode_with_error_round_trips_through_json() -> None:
    """Tests a failed Episode can be serialized and read back."""
    task = Task(instruction="mock instruction")
    ep = Episode(task=task, rollout="", error=ValueError("boom"))

    restored = Episode.model_validate_json(ep.model_dump_json())

    assert restored.error == "ValueError: boom"
    assert Episode.model_validate_json(restored.model_dump_json()).error == (
        "ValueError: boom"
    )


def test_episode_without_error_serializes_none() -> None:
    """Tests a successful Episode serializes ``error`` as null."""
    task = Task(instruction="mock instruction")
    ep = Episode(task=task, rollout="")

    assert Episode.model_validate_json(ep.model_dump_json()).error is None
