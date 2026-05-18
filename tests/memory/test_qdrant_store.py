from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import QdrantClient

from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.memory import Episode
from llm_agents_from_scratch.memory.qdrant_store import QdrantMemoryStore


@pytest.fixture
def episode() -> Episode:
    task = Task(instruction="look up pikachu")
    return Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="pikachu is electric type"),
    )


@pytest.fixture
def mock_client():
    with patch(
        "llm_agents_from_scratch.memory.qdrant_store.QdrantClient",
    ) as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


def test_init(mock_client: MagicMock) -> None:
    mock_client.collection_exists.return_value = False
    QdrantMemoryStore()
    mock_client.set_model.assert_called_once_with("BAAI/bge-small-en-v1.5")
    mock_client.create_collection.assert_called_once()


def test_init_skips_create_if_collection_exists(
    mock_client: MagicMock,
) -> None:
    mock_client.collection_exists.return_value = True
    QdrantMemoryStore()
    mock_client.create_collection.assert_not_called()


def test_init_custom_params(mock_client: MagicMock) -> None:
    store = QdrantMemoryStore(
        collection_name="my_eps",
        embedding_model="BAAI/bge-large-en-v1.5",
    )
    mock_client.set_model.assert_called_once_with("BAAI/bge-large-en-v1.5")
    assert store._collection == "my_eps"


def test_init_custom_client() -> None:
    custom_client = MagicMock(spec=QdrantClient)
    store = QdrantMemoryStore(client=custom_client)
    assert store._client is custom_client
    custom_client.set_model.assert_called_once()


async def test_write(mock_client: MagicMock, episode: Episode) -> None:
    store = QdrantMemoryStore()
    await store.write(episode)

    mock_client.add.assert_called_once()
    kw = mock_client.add.call_args.kwargs
    assert kw["collection_name"] == "episodes"
    assert episode.task.instruction in kw["documents"][0]
    assert episode.result.content in kw["documents"][0]
    assert kw["ids"] == [episode.task.id_]
    assert "episode_json" in kw["metadata"][0]
    assert "completed_at" in kw["metadata"][0]


async def test_count(mock_client: MagicMock) -> None:
    expected = 7
    mock_client.count.return_value.count = expected
    store = QdrantMemoryStore()
    assert await store.count() == expected


async def test_read_recent_empty(mock_client: MagicMock) -> None:
    mock_client.count.return_value.count = 0
    store = QdrantMemoryStore()
    assert await store.read_recent(3) == []


async def test_read_recent(mock_client: MagicMock, episode: Episode) -> None:
    mock_client.count.return_value.count = 1
    record = MagicMock()
    record.payload = {
        "episode_json": episode.model_dump_json(),
        "completed_at": episode.completed_at.timestamp(),
    }
    mock_client.scroll.return_value = ([record], None)

    store = QdrantMemoryStore()
    result = await store.read_recent(5)

    assert len(result) == 1
    assert result[0].task.instruction == episode.task.instruction


async def test_read_recent_sorted_newest_first(
    mock_client: MagicMock,
) -> None:
    older_task = Task(instruction="older task")
    newer_task = Task(instruction="newer task")
    older = Episode(
        task=older_task,
        rollout="",
        result=TaskResult(task_id=older_task.id_, content="r"),
        completed_at=datetime(2025, 1, 1),
    )
    newer = Episode(
        task=newer_task,
        rollout="",
        result=TaskResult(task_id=newer_task.id_, content="r"),
        completed_at=datetime(2025, 6, 1),
    )

    mock_client.count.return_value.count = 2
    records = []
    for ep in [older, newer]:
        rec = MagicMock()
        rec.payload = {
            "episode_json": ep.model_dump_json(),
            "completed_at": ep.completed_at.timestamp(),
        }
        records.append(rec)
    mock_client.scroll.return_value = (records, None)

    store = QdrantMemoryStore()
    result = await store.read_recent(2)

    assert result[0].task.instruction == "newer task"
    assert result[1].task.instruction == "older task"


async def test_read_recent_respects_n_limit(mock_client: MagicMock) -> None:
    episodes = []
    for i in range(5):
        t = Task(instruction=f"task {i}")
        episodes.append(
            Episode(
                task=t,
                rollout="",
                result=TaskResult(task_id=t.id_, content=f"result {i}"),
                completed_at=datetime(2025, 1, i + 1),
            ),
        )

    mock_client.count.return_value.count = 5
    records = []
    for ep in episodes:
        rec = MagicMock()
        rec.payload = {
            "episode_json": ep.model_dump_json(),
            "completed_at": ep.completed_at.timestamp(),
        }
        records.append(rec)
    mock_client.scroll.return_value = (records, None)

    n = 3
    store = QdrantMemoryStore()
    result = await store.read_recent(n)

    assert len(result) == n


async def test_search(mock_client: MagicMock, episode: Episode) -> None:
    mock_result = MagicMock()
    mock_result.metadata = {"episode_json": episode.model_dump_json()}
    mock_client.query.return_value = [mock_result]

    store = QdrantMemoryStore()
    results = await store.search("electric type pokemon", k=3)

    mock_client.query.assert_called_once_with(
        collection_name="episodes",
        query_text="electric type pokemon",
        limit=3,
    )
    assert len(results) == 1
    assert results[0].task.instruction == episode.task.instruction


async def test_search_empty(mock_client: MagicMock) -> None:
    mock_client.query.return_value = []
    store = QdrantMemoryStore()
    results = await store.search("anything", k=5)
    assert results == []


async def test_summary_empty(mock_client: MagicMock) -> None:
    mock_client.count.return_value.count = 0
    store = QdrantMemoryStore()
    summary = await store.summary()
    assert "QdrantMemoryStore" in summary
    assert "0" in summary


async def test_summary_with_episodes(
    mock_client: MagicMock,
    episode: Episode,
) -> None:
    ep_json = episode.model_dump_json()
    ts = episode.completed_at.timestamp()
    record = MagicMock()
    record.payload = {"episode_json": ep_json, "completed_at": ts}

    mock_client.count.return_value.count = 1
    mock_client.scroll.return_value = ([record], None)

    store = QdrantMemoryStore()
    summary = await store.summary()

    assert "QdrantMemoryStore" in summary
    assert "1" in summary
    assert episode.task.instruction[:20] in summary
