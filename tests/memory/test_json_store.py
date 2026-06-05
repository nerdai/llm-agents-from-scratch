"""Unit tests for JSONMemoryStore."""

import json
from pathlib import Path

import pytest

from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.memory import Episode, RecallMode
from llm_agents_from_scratch.errors import EpisodeNotFoundError
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
    store = JSONMemoryStore(dir=tmp_path)
    assert store._episodes == []


def test_init_loads_existing_episodes(tmp_path: Path) -> None:
    """Tests store loads episodes from an existing JSONL file on init."""
    ep = make_episode()
    (tmp_path / "episodes.jsonl").write_text(ep.model_dump_json() + "\n")

    store = JSONMemoryStore(dir=tmp_path)
    assert len(store._episodes) == 1
    assert store._episodes[0].task.instruction == ep.task.instruction


@pytest.mark.asyncio
async def test_write_persists_to_disk(tmp_path: Path) -> None:
    """Tests write appends one JSON line to the JSONL file."""
    store = JSONMemoryStore(dir=tmp_path)
    ep = make_episode()

    await store.write(ep)

    assert store.path.exists()
    lines = [ln for ln in store.path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["task"]["instruction"] == ep.task.instruction  # noqa: E501


@pytest.mark.asyncio
async def test_custom_filename(tmp_path: Path) -> None:
    """Tests store uses the caller-supplied filename within the directory."""
    store = JSONMemoryStore(dir=tmp_path, filename="custom.jsonl")
    assert store.path == tmp_path / "custom.jsonl"


@pytest.mark.asyncio
async def test_count(tmp_path: Path) -> None:
    """Tests count returns the number of stored episodes."""
    store = JSONMemoryStore(dir=tmp_path)

    assert await store.count() == 0
    await store.write(make_episode("task 1"))
    await store.write(make_episode("task 2"))
    assert await store.count() == 2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_write_multiple_episodes(tmp_path: Path) -> None:
    """Tests multiple writes accumulate on disk."""
    store = JSONMemoryStore(dir=tmp_path)

    await store.write(make_episode("task 1"))
    await store.write(make_episode("task 2"))
    await store.write(make_episode("task 3"))

    assert len(store._episodes) == 3  # noqa: PLR2004


@pytest.mark.asyncio
async def test_reload_restores_written_episodes(tmp_path: Path) -> None:
    """Tests a new store instance loads episodes written by a previous one."""
    store = JSONMemoryStore(dir=tmp_path)
    await store.write(make_episode("persisted task"))

    store2 = JSONMemoryStore(dir=tmp_path)
    assert len(store2._episodes) == 1
    assert store2._episodes[0].task.instruction == "persisted task"


@pytest.mark.asyncio
async def test_read_recent_returns_n_most_recent(tmp_path: Path) -> None:
    """Tests read_recent returns episodes ordered newest-first, capped at n."""
    store = JSONMemoryStore(dir=tmp_path)
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
    store = JSONMemoryStore(dir=tmp_path)
    await store.write(make_episode())

    recent = await store.read_recent(10)
    assert len(recent) == 1


@pytest.mark.asyncio
async def test_search_returns_recent_by_default(tmp_path: Path) -> None:
    """Tests search delegates to read_recent when recall_mode='recent'."""
    store = JSONMemoryStore(dir=tmp_path, max_results=2)
    ep1 = make_episode("first")
    ep2 = make_episode("second")
    ep3 = make_episode("third")
    await store.write(ep1)
    await store.write(ep2)
    await store.write(ep3)

    results = await store.search("ignored query")

    assert len(results) == 2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_search_raises_not_implemented_when_search_mode(
    tmp_path: Path,
) -> None:
    """Tests search raises NotImplementedError when recall_mode='search'."""
    store = JSONMemoryStore(dir=tmp_path, recall_mode=RecallMode.SEARCH)

    with pytest.raises(NotImplementedError):
        await store.search("query")


@pytest.mark.asyncio
async def test_summary_empty(tmp_path: Path) -> None:
    """Tests summary reports zero episodes when store is empty."""
    store = JSONMemoryStore(dir=tmp_path)

    summary = await store.summary()

    assert "JSONMemoryStore" in summary
    assert "0" in summary


@pytest.mark.asyncio
async def test_summary_includes_newest_and_oldest(tmp_path: Path) -> None:
    """Tests summary includes newest and oldest episode instructions."""
    store = JSONMemoryStore(dir=tmp_path)
    await store.write(make_episode("oldest task"))
    await store.write(make_episode("newest task"))

    summary = await store.summary()

    assert "2" in summary
    assert "oldest task" in summary
    assert "newest task" in summary


# --- delete ---


@pytest.mark.asyncio
async def test_delete_removes_episode(tmp_path: Path) -> None:
    store = JSONMemoryStore(dir=tmp_path)
    ep = make_episode()
    await store.write(ep)

    await store.delete(ep.id_)

    assert await store.count() == 0
    assert store.path.read_text() == ""


@pytest.mark.asyncio
async def test_delete_rewrites_file_keeping_other_episodes(
    tmp_path: Path,
) -> None:
    store = JSONMemoryStore(dir=tmp_path)
    ep1 = make_episode("keep me")
    ep2 = make_episode("delete me")
    await store.write(ep1)
    await store.write(ep2)

    await store.delete(ep2.id_)

    assert await store.count() == 1
    assert store._episodes[0].id_ == ep1.id_


@pytest.mark.asyncio
async def test_delete_warns_when_id_not_found(tmp_path: Path) -> None:
    store = JSONMemoryStore(dir=tmp_path)

    with pytest.raises(EpisodeNotFoundError):
        await store.delete("nonexistent-id")


# --- update ---


@pytest.mark.asyncio
async def test_update_replaces_episode(tmp_path: Path) -> None:
    store = JSONMemoryStore(dir=tmp_path)
    ep = make_episode("original")
    await store.write(ep)

    updated = Episode(
        id_=ep.id_,
        task=ep.task,
        rollout="",
        result=TaskResult(task_id=ep.task.id_, content="updated content"),
    )
    await store.update(updated)

    episodes = await store.read_recent(1)
    assert episodes[0].result.content == "updated content"


@pytest.mark.asyncio
async def test_update_persists_to_file(tmp_path: Path) -> None:
    store = JSONMemoryStore(dir=tmp_path)
    ep = make_episode("original")
    await store.write(ep)

    updated = Episode(
        id_=ep.id_,
        task=ep.task,
        rollout="",
        result=TaskResult(task_id=ep.task.id_, content="updated"),
    )
    await store.update(updated)

    reloaded = JSONMemoryStore(dir=tmp_path)
    episodes = await reloaded.read_recent(1)
    assert episodes[0].result.content == "updated"


@pytest.mark.asyncio
async def test_update_warns_when_id_not_found(tmp_path: Path) -> None:
    store = JSONMemoryStore(dir=tmp_path)
    ep = make_episode()

    with pytest.raises(EpisodeNotFoundError):
        await store.update(ep)
