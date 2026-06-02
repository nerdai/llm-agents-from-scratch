"""Unit tests for RecencyMemory recipe."""

from pathlib import Path

import pytest

from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.memory import Episode, RecallMode
from llm_agents_from_scratch.memory import Memory, recency_memory
from llm_agents_from_scratch.memory_stores.json import JSONMemoryStore


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


def test_returns_memory_instance(tmp_path: Path) -> None:
    memory = recency_memory(path=tmp_path)
    assert isinstance(memory, Memory)


def test_store_is_json(tmp_path: Path) -> None:
    memory = recency_memory(path=tmp_path)
    assert isinstance(memory.store, JSONMemoryStore)


def test_store_recall_mode_is_recent(tmp_path: Path) -> None:
    memory = recency_memory(path=tmp_path)
    assert memory.store.recall_mode == RecallMode.RECENT


def test_max_results_set_from_n(tmp_path: Path) -> None:
    memory = recency_memory(path=tmp_path, n=7)
    assert memory.store.max_results == 7  # noqa: PLR2004


def test_default_key_fn_uses_instruction(tmp_path: Path) -> None:
    memory = recency_memory(path=tmp_path)
    ep = make_episode("look up pikachu")
    assert memory.key_fn(ep) == "look up pikachu"


def test_no_metadata_fns_by_default(tmp_path: Path) -> None:
    memory = recency_memory(path=tmp_path)
    assert memory.metadata_fns == {}


@pytest.mark.asyncio
async def test_recall_returns_formatted_episodes(tmp_path: Path) -> None:
    memory = recency_memory(path=tmp_path, n=2)
    ep1 = make_episode("task one", "result one")
    ep2 = make_episode("task two", "result two")
    await memory.record(ep1)
    await memory.record(ep2)

    result = await memory.recall(Task(instruction="new task"))

    assert str(ep1) in result
    assert str(ep2) in result


@pytest.mark.asyncio
async def test_recall_returns_empty_string_when_no_episodes(
    tmp_path: Path,
) -> None:
    memory = recency_memory(path=tmp_path)
    result = await memory.recall(Task(instruction="new task"))
    assert result == ""
