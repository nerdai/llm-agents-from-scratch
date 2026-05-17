from datetime import datetime
from unittest.mock import MagicMock

from llm_agents_from_scratch.data_structures import (
    Task,
    TaskResult,
)
from llm_agents_from_scratch.data_structures.memory import Episode


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
