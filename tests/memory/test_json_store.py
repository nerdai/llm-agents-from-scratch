"""Unit tests for JSONMemoryStore."""

import json
from pathlib import Path

import pytest

from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.memory import Episode
from llm_agents_from_scratch.memory import JSONMemoryStore


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


def test_init_empty_when_file_missing(tmp_path: Path) -> None:
    """Tests store starts empty when backing file does not exist."""
    store = JSONMemoryStore(path=tmp_path / "episodes.jsonl")
    assert store._episodes == []


def test_init_loads_existing_episodes(tmp_path: Path) -> None:
    """Tests store loads episodes from an existing JSONL file on init."""
    path = tmp_path / "episodes.jsonll"
    ep = make_episode()
    path.write_text(ep.model_dump_json() + "\n")

    store = JSONMemoryStore(path=path)
    assert len(store._episodes) == 1
    assert store._episodes[0].task.instruction == ep.task.instruction


@pytest.mark.asyncio
async def test_write_persists_to_disk(tmp_path: Path) -> None:
    """Tests write appends one JSON line to the JSONL file."""
    path = tmp_path / "episodes.jsonll"
    store = JSONMemoryStore(path=path)
    ep = make_episode()

    await store.write(ep)

    assert path.exists()
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["task"]["instruction"] == ep.task.instruction  # noqa: E501


@pytest.mark.asyncio
async def test_write_multiple_episodes(tmp_path: Path) -> None:
    """Tests multiple writes accumulate on disk."""
    store = JSONMemoryStore(path=tmp_path / "episodes.jsonl")

    await store.write(make_episode("task 1"))
    await store.write(make_episode("task 2"))
    await store.write(make_episode("task 3"))

    assert len(store._episodes) == 3  # noqa: PLR2004


@pytest.mark.asyncio
async def test_reload_restores_written_episodes(tmp_path: Path) -> None:
    """Tests a new store instance loads episodes written by a previous one."""
    path = tmp_path / "episodes.jsonl"
    store = JSONMemoryStore(path=path)
    await store.write(make_episode("persisted task"))

    store2 = JSONMemoryStore(path=path)
    assert len(store2._episodes) == 1
    assert store2._episodes[0].task.instruction == "persisted task"


@pytest.mark.asyncio
async def test_read_recent_returns_n_most_recent(tmp_path: Path) -> None:
    """Tests read_recent returns episodes ordered newest-first, capped at n."""
    store = JSONMemoryStore(path=tmp_path / "episodes.jsonl")
    await store.write(make_episode("oldest"))
    await store.write(make_episode("middle"))
    await store.write(make_episode("newest"))

    recent = await store.read_recent(2)

    assert len(recent) == 2  # noqa: PLR2004
    assert recent[0].task.instruction == "newest"
    assert recent[1].task.instruction == "middle"


@pytest.mark.asyncio
async def test_read_recent_returns_all_when_n_exceeds_count(
    tmp_path: Path,
) -> None:
    """Tests read_recent returns all episodes when n > stored count."""
    store = JSONMemoryStore(path=tmp_path / "episodes.jsonl")
    await store.write(make_episode())

    recent = await store.read_recent(10)
    assert len(recent) == 1


@pytest.mark.asyncio
async def test_search_raises_not_implemented(tmp_path: Path) -> None:
    """Tests search raises NotImplementedError."""
    store = JSONMemoryStore(path=tmp_path / "episodes.jsonl")

    with pytest.raises(NotImplementedError):
        await store.search("query", k=3)
