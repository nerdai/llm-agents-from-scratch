"""Unit tests for RecencyMemory."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_agents_from_scratch.base.memory import BaseMemoryStore
from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.memory import Episode
from llm_agents_from_scratch.memory import JSONMemoryStore, RecencyMemory


def make_episode(
    instruction: str = "do something",
    content: str = "done",
) -> Episode:
    task = Task(instruction=instruction)
    return Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content=content),
    )


def test_init_defaults() -> None:
    """Tests RecencyMemory initialises with n=3 by default."""
    store = MagicMock(spec=BaseMemoryStore)
    memory = RecencyMemory(store=store)

    assert memory.store is store
    assert memory.n == 3  # noqa: PLR2004


def test_init_custom_n() -> None:
    """Tests RecencyMemory accepts a custom n."""
    store = MagicMock(spec=BaseMemoryStore)
    memory = RecencyMemory(store=store, n=10)

    assert memory.n == 10  # noqa: PLR2004


@pytest.mark.asyncio
async def test_recall_returns_formatted_episodes() -> None:
    """Tests recall returns newline-joined episode strings."""
    ep1 = make_episode("task one", "result one")
    ep2 = make_episode("task two", "result two")

    store = AsyncMock(spec=BaseMemoryStore)
    store.read_recent.return_value = [ep2, ep1]
    memory = RecencyMemory(store=store, n=2)

    result = await memory.recall(Task(instruction="new task"))

    store.read_recent.assert_awaited_once_with(2)
    assert str(ep2) in result
    assert str(ep1) in result


@pytest.mark.asyncio
async def test_recall_returns_empty_string_when_no_episodes() -> None:
    """Tests recall returns empty string when the store is empty."""
    store = AsyncMock(spec=BaseMemoryStore)
    store.read_recent.return_value = []
    memory = RecencyMemory(store=store)

    result = await memory.recall(Task(instruction="new task"))

    assert result == ""


@pytest.mark.asyncio
async def test_record_delegates_to_store(tmp_path: Path) -> None:
    """Tests record writes the episode to the store."""
    store = AsyncMock(spec=BaseMemoryStore)
    memory = RecencyMemory(store=store)
    ep = make_episode()

    await memory.record(ep)

    store.write.assert_awaited_once_with(ep)


@pytest.mark.asyncio
async def test_summary(tmp_path: Path) -> None:
    """Tests summary returns episode count and recall window."""
    store = JSONMemoryStore(dir=tmp_path)
    memory = RecencyMemory(store=store, n=5)

    await store.write(make_episode("task 1"))
    await store.write(make_episode("task 2"))

    summary = await memory.summary()

    assert "2" in summary
    assert "5" in summary
