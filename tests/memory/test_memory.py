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


def make_store(episodes: list[Episode] | None = None) -> AsyncMock:
    store = AsyncMock(spec=BaseMemoryStore)
    store.max_results = 5
    store.search.return_value = episodes or []
    return store


# --- construction ---


def test_init_stores_attributes() -> None:
    store = MagicMock(spec=BaseMemoryStore)
    transform = AsyncMock()

    memory = Memory(store=store, transformations=[transform])

    assert memory.store is store
    assert memory.transformations == [transform]


def test_init_default_transformations() -> None:
    store = MagicMock(spec=BaseMemoryStore)

    memory = Memory(store=store)

    assert memory.transformations == []


# --- recall ---


@pytest.mark.asyncio
async def test_recall_formats_episodes() -> None:
    ep1 = make_episode("task one", "result one")
    ep2 = make_episode("task two", "result two")
    store = make_store([ep1, ep2])

    memory = Memory(store=store)
    task = Task(instruction="new task")

    result = await memory.recall(task)

    store.search.assert_awaited_once_with(task.instruction)
    assert str(ep1) in result
    assert str(ep2) in result


@pytest.mark.asyncio
async def test_recall_returns_empty_string_when_no_episodes() -> None:
    store = make_store([])

    memory = Memory(store=store)

    result = await memory.recall(Task(instruction="anything"))

    assert result == ""


# --- record ---


@pytest.mark.asyncio
async def test_record_writes_to_store_without_transformations() -> None:
    store = make_store()
    memory = Memory(store=store)
    ep = make_episode()

    await memory.record(ep)

    store.write.assert_awaited_once_with(ep)


@pytest.mark.asyncio
async def test_record_applies_transformations_in_order() -> None:
    store = make_store()
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

    memory = Memory(store=store, transformations=[transform_a, transform_b])
    ep = make_episode()

    await memory.record(ep)

    assert call_order == ["a", "b"]
    store.write.assert_awaited_once()
    written_ep: Episode = store.write.call_args[0][0]
    assert written_ep.additional_data == {"step": "b"}


@pytest.mark.asyncio
async def test_record_passes_transformed_episode_to_store() -> None:
    store = make_store()

    async def add_annotation(ep: Episode) -> Episode:
        ep.additional_data = {"note": "annotated"}
        return ep

    memory = Memory(store=store, transformations=[add_annotation])
    ep = make_episode()

    await memory.record(ep)

    written_ep: Episode = store.write.call_args[0][0]
    assert written_ep.additional_data == {"note": "annotated"}


# --- delete / update ---


@pytest.mark.asyncio
async def test_delete_raises_not_implemented() -> None:
    memory = Memory(store=MagicMock())

    with pytest.raises(NotImplementedError):
        await memory.delete("some-id")


@pytest.mark.asyncio
async def test_update_raises_not_implemented() -> None:
    memory = Memory(store=MagicMock())

    with pytest.raises(NotImplementedError):
        await memory.update(make_episode())
