"""Unit tests for SimilarityMemory."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_agents_from_scratch.base.memory_store import BaseMemoryStore
from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.memory import Episode
from llm_agents_from_scratch.memory import SimilarityMemory


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
    """Tests SimilarityMemory initialises with k=3 and empty kwargs."""
    store = MagicMock(spec=BaseMemoryStore)
    memory = SimilarityMemory(store=store)

    assert memory.store is store
    assert memory.k == 3  # noqa: PLR2004
    assert memory.search_kwargs == {}


def test_init_custom_params() -> None:
    """Tests SimilarityMemory accepts custom k and search_kwargs."""
    store = MagicMock(spec=BaseMemoryStore)
    memory = SimilarityMemory(
        store=store,
        k=5,
        search_kwargs={"score_threshold": 0.8},
    )

    assert memory.k == 5  # noqa: PLR2004
    assert memory.search_kwargs == {"score_threshold": 0.8}


@pytest.mark.asyncio
async def test_recall_returns_formatted_episodes() -> None:
    """Tests recall returns newline-joined episode strings."""
    ep1 = make_episode("task one", "result one")
    ep2 = make_episode("task two", "result two")

    store = AsyncMock(spec=BaseMemoryStore)
    store.search.return_value = [ep1, ep2]
    memory = SimilarityMemory(store=store, k=2)

    result = await memory.recall(Task(instruction="relevant query"))

    store.search.assert_awaited_once_with(query="relevant query")
    assert str(ep1) in result
    assert str(ep2) in result


@pytest.mark.asyncio
async def test_recall_forwards_search_kwargs() -> None:
    """Tests recall forwards search_kwargs to store.search."""
    store = AsyncMock(spec=BaseMemoryStore)
    store.search.return_value = []
    memory = SimilarityMemory(
        store=store,
        k=3,
        search_kwargs={"score_threshold": 0.9},
    )

    await memory.recall(Task(instruction="query"))

    store.search.assert_awaited_once_with(
        query="query",
        score_threshold=0.9,
    )


@pytest.mark.asyncio
async def test_recall_returns_empty_string_when_no_episodes() -> None:
    """Tests recall returns empty string when search finds nothing."""
    store = AsyncMock(spec=BaseMemoryStore)
    store.search.return_value = []
    memory = SimilarityMemory(store=store)

    result = await memory.recall(Task(instruction="query"))

    assert result == ""


@pytest.mark.asyncio
async def test_record_delegates_to_store() -> None:
    """Tests record writes the episode to the store."""
    store = AsyncMock(spec=BaseMemoryStore)
    memory = SimilarityMemory(store=store)
    ep = make_episode()

    await memory.record(ep)

    store.write.assert_awaited_once_with(ep)


@pytest.mark.asyncio
async def test_summary_delegates_to_store() -> None:
    """Tests summary includes k and delegates to store.summary."""
    store = AsyncMock(spec=BaseMemoryStore)
    store.summary.return_value = "store details"
    memory = SimilarityMemory(store=store, k=4)

    summary = await memory.summary()

    store.summary.assert_awaited_once()
    assert "4" in summary
    assert "SimilarityMemory" in summary
    assert "store details" in summary
