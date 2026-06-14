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
    metadata_fn = AsyncMock(return_value="value")

    memory = Memory(
        store=store,
        metadata_fns={"note": metadata_fn},
    )

    assert memory.store is store
    assert memory.metadata_fns == {"note": metadata_fn}


def test_init_default_metadata_fns() -> None:
    store = MagicMock(spec=BaseMemoryStore)

    memory = Memory(store=store)

    assert memory.metadata_fns == {}


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
async def test_record_writes_episode() -> None:
    store = make_store()
    ep = make_episode("summarise doc")
    memory = Memory(store=store)

    await memory.record(ep)

    store.write.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_runs_metadata_fns_concurrently() -> None:
    store = make_store()
    ep = make_episode()

    async def reflect(episode: Episode) -> str:
        return "a lesson"

    def tag(episode: Episode) -> str:
        return "important"

    memory = Memory(
        store=store,
        metadata_fns={"reflection": reflect, "tag": tag},
    )

    await memory.record(ep)

    written: Episode = store.write.call_args.args[0]
    assert written.metadata["reflection"] == "a lesson"
    assert written.metadata["tag"] == "important"
    store.write.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_does_not_mutate_original_episode() -> None:
    store = make_store()
    ep = make_episode()

    memory = Memory(
        store=store,
        metadata_fns={"note": lambda _: "enriched"},
    )

    await memory.record(ep)

    assert ep.metadata == {}


@pytest.mark.asyncio
async def test_record_all_metadata_fns_called() -> None:
    store = make_store()
    ep = make_episode()
    fn_a = AsyncMock(return_value="val_a")
    fn_b = AsyncMock(return_value="val_b")

    memory = Memory(
        store=store,
        metadata_fns={"a": fn_a, "b": fn_b},
    )

    await memory.record(ep)

    fn_a.assert_awaited_once()
    fn_b.assert_awaited_once()


# --- summary ---


@pytest.mark.asyncio
async def test_summary_includes_store_summary_and_config() -> None:
    store = AsyncMock()
    store.summary = AsyncMock(return_value="store summary")
    memory = Memory(store=store)

    result = await memory.summary()

    store.summary.assert_called_once()
    assert "store summary" in result
    assert "metadata_fns: none" in result


@pytest.mark.asyncio
async def test_summary_lists_metadata_fn_keys() -> None:
    store = AsyncMock()
    store.summary = AsyncMock(return_value="store summary")
    memory = Memory(
        store=store,
        metadata_fns={"reflection": AsyncMock(), "tag": AsyncMock()},
    )

    result = await memory.summary()

    assert "reflection" in result
    assert "tag" in result


# --- delete / update ---


@pytest.mark.asyncio
async def test_delete_delegates_to_store() -> None:
    store = AsyncMock()
    memory = Memory(store=store)

    await memory.delete("some-id")

    store.delete.assert_awaited_once_with("some-id")


@pytest.mark.asyncio
async def test_update_delegates_to_store() -> None:
    store = AsyncMock()
    memory = Memory(store=store)
    ep = make_episode()

    await memory.update(ep)

    store.update.assert_awaited_once_with(ep)
