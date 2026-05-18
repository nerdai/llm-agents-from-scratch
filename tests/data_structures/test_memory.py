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
