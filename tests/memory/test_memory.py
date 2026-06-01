"""Unit tests for Memory."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_agents_from_scratch.base.memory import BaseMemoryStore
from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.memory import Episode
from llm_agents_from_scratch.memory import Memory


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


# --- construction ---


def test_init_stores_attributes() -> None:
    store = MagicMock(spec=BaseMemoryStore)
    recall_fn = AsyncMock(return_value=[])
    transform = AsyncMock()

    memory = Memory(
        store=store,
        recall_fn=recall_fn,
        transformations=[transform],
    )

    assert memory.store is store
    assert memory.recall_fn is recall_fn
    assert memory.transformations == [transform]


def test_init_default_transformations() -> None:
    store = MagicMock(spec=BaseMemoryStore)
    recall_fn = AsyncMock(return_value=[])

    memory = Memory(store=store, recall_fn=recall_fn)

    assert memory.transformations == []


# --- recall ---


@pytest.mark.asyncio
async def test_recall_formats_episodes() -> None:
    ep1 = make_episode("task one", "result one")
    ep2 = make_episode("task two", "result two")
    recall_fn = AsyncMock(return_value=[ep1, ep2])

    memory = Memory(store=MagicMock(), recall_fn=recall_fn)
    task = Task(instruction="new task")

    result = await memory.recall(task)

    recall_fn.assert_awaited_once_with(task)
    assert str(ep1) in result
    assert str(ep2) in result


@pytest.mark.asyncio
async def test_recall_returns_empty_string_when_no_episodes() -> None:
    recall_fn = AsyncMock(return_value=[])

    memory = Memory(store=MagicMock(), recall_fn=recall_fn)

    result = await memory.recall(Task(instruction="anything"))

    assert result == ""


# --- record ---


@pytest.mark.asyncio
async def test_record_writes_to_store_without_transformations() -> None:
    store = AsyncMock(spec=BaseMemoryStore)
    memory = Memory(store=store, recall_fn=AsyncMock(return_value=[]))
    ep = make_episode()

    await memory.record(ep)

    store.write.assert_awaited_once_with(ep)


@pytest.mark.asyncio
async def test_record_applies_transformations_in_order() -> None:
    store = AsyncMock(spec=BaseMemoryStore)
    call_order: list[str] = []

    async def transform_a(ep: Episode) -> Episode:
        call_order.append("a")
        ep.additional_data = {"step": "a"}
        return ep

    async def transform_b(ep: Episode) -> Episode:
        call_order.append("b")
        assert ep.additional_data == {"step": "a"}
        ep.additional_data = {"step": "b"}
        return ep

    memory = Memory(
        store=store,
        recall_fn=AsyncMock(return_value=[]),
        transformations=[transform_a, transform_b],
    )
    ep = make_episode()

    await memory.record(ep)

    assert call_order == ["a", "b"]
    store.write.assert_awaited_once()
    written_ep: Episode = store.write.call_args[0][0]
    assert written_ep.additional_data == {"step": "b"}


@pytest.mark.asyncio
async def test_record_passes_transformed_episode_to_store() -> None:
    store = AsyncMock(spec=BaseMemoryStore)

    async def add_annotation(ep: Episode) -> Episode:
        ep.additional_data = {"note": "annotated"}
        return ep

    memory = Memory(
        store=store,
        recall_fn=AsyncMock(return_value=[]),
        transformations=[add_annotation],
    )
    ep = make_episode()

    await memory.record(ep)

    written_ep: Episode = store.write.call_args[0][0]
    assert written_ep.additional_data == {"note": "annotated"}


# --- delete / update ---


@pytest.mark.asyncio
async def test_delete_raises_not_implemented() -> None:
    memory = Memory(store=MagicMock(), recall_fn=AsyncMock(return_value=[]))

    with pytest.raises(NotImplementedError):
        await memory.delete("some-id")


@pytest.mark.asyncio
async def test_update_raises_not_implemented() -> None:
    memory = Memory(store=MagicMock(), recall_fn=AsyncMock(return_value=[]))

    with pytest.raises(NotImplementedError):
        await memory.update(make_episode())
