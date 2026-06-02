"""Unit tests for Memory."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_agents_from_scratch.base.memory_store import BaseMemoryStore
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
    key_fn = lambda ep: ep.task.instruction  # noqa: E731
    metadata_fn = AsyncMock(return_value="value")

    memory = Memory(
        store=store,
        key_fn=key_fn,
        metadata_fns={"note": metadata_fn},
    )

    assert memory.store is store
    assert memory.key_fn is key_fn
    assert memory.metadata_fns == {"note": metadata_fn}


def test_init_default_metadata_fns() -> None:
    store = MagicMock(spec=BaseMemoryStore)

    memory = Memory(store=store, key_fn=lambda ep: ep.task.instruction)

    assert memory.metadata_fns == {}


# --- recall ---


@pytest.mark.asyncio
async def test_recall_formats_episodes() -> None:
    ep1 = make_episode("task one", "result one")
    ep2 = make_episode("task two", "result two")
    store = make_store([ep1, ep2])

    memory = Memory(store=store, key_fn=lambda ep: ep.task.instruction)
    task = Task(instruction="new task")

    result = await memory.recall(task)

    store.search.assert_awaited_once_with(task.instruction)
    assert str(ep1) in result
    assert str(ep2) in result


@pytest.mark.asyncio
async def test_recall_returns_empty_string_when_no_episodes() -> None:
    store = make_store([])

    memory = Memory(store=store, key_fn=lambda ep: ep.task.instruction)

    result = await memory.recall(Task(instruction="anything"))

    assert result == ""


# --- record ---


@pytest.mark.asyncio
async def test_record_writes_episode_with_key() -> None:
    store = make_store()
    ep = make_episode("summarise doc")
    memory = Memory(store=store, key_fn=lambda ep: ep.task.instruction)

    await memory.record(ep)

    store.write.assert_awaited_once_with(ep, "summarise doc")


@pytest.mark.asyncio
async def test_record_runs_metadata_fns_concurrently() -> None:
    store = make_store()
    ep = make_episode()

    async def reflect(ep: Episode) -> str:
        return "a lesson"

    def tag(ep: Episode) -> str:
        return "important"

    memory = Memory(
        store=store,
        key_fn=lambda ep: ep.task.instruction,
        metadata_fns={"reflection": reflect, "tag": tag},
    )

    await memory.record(ep)

    assert ep.metadata["reflection"] == "a lesson"
    assert ep.metadata["tag"] == "important"
    store.write.assert_awaited_once_with(ep, ep.task.instruction)


@pytest.mark.asyncio
async def test_record_all_metadata_fns_called() -> None:
    store = make_store()
    ep = make_episode()
    fn_a = AsyncMock(return_value="val_a")
    fn_b = AsyncMock(return_value="val_b")

    memory = Memory(
        store=store,
        key_fn=lambda ep: ep.task.instruction,
        metadata_fns={"a": fn_a, "b": fn_b},
    )

    await memory.record(ep)

    fn_a.assert_awaited_once_with(ep)
    fn_b.assert_awaited_once_with(ep)


# --- delete / update ---


@pytest.mark.asyncio
async def test_delete_raises_not_implemented() -> None:
    memory = Memory(store=MagicMock(), key_fn=lambda ep: ep.task.instruction)

    with pytest.raises(NotImplementedError):
        await memory.delete("some-id")


@pytest.mark.asyncio
async def test_update_raises_not_implemented() -> None:
    memory = Memory(store=MagicMock(), key_fn=lambda ep: ep.task.instruction)

    with pytest.raises(NotImplementedError):
        await memory.update(make_episode())
