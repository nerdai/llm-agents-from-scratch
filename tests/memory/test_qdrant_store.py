from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import AsyncQdrantClient, models

from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.memory import Episode, RecallMode
from llm_agents_from_scratch.errors import EpisodeNotFoundError
from llm_agents_from_scratch.memory_stores.qdrant.store import QdrantMemoryStore


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
        "llm_agents_from_scratch.memory_stores.qdrant.store.AsyncQdrantClient",
    ) as mock_cls:
        instance = MagicMock()
        instance.embedding_model_name = "BAAI/bge-small-en-v1.5"
        instance.get_vector_field_name.return_value = "fast-bge-small-en-v1.5"
        instance.collection_exists = AsyncMock(return_value=True)
        instance.create_collection = AsyncMock()
        instance.upsert = AsyncMock()
        instance.count = AsyncMock(return_value=MagicMock(count=0))
        instance.scroll = AsyncMock(return_value=([], None))
        instance.retrieve = AsyncMock(return_value=[])
        instance.delete = AsyncMock()
        instance.query_points = AsyncMock(
            return_value=MagicMock(points=[]),
        )
        mock_cls.return_value = instance
        yield instance


async def test_ensure_collection_creates_when_missing(
    mock_client: MagicMock,
) -> None:
    mock_client.collection_exists = AsyncMock(return_value=False)
    store = QdrantMemoryStore()
    await store._ensure_collection()
    mock_client.create_collection.assert_called_once()


async def test_ensure_collection_skips_create_if_exists(
    mock_client: MagicMock,
) -> None:
    mock_client.collection_exists = AsyncMock(return_value=True)
    store = QdrantMemoryStore()
    await store._ensure_collection()
    mock_client.create_collection.assert_not_called()


async def test_ensure_collection_called_only_once(
    mock_client: MagicMock,
) -> None:
    store = QdrantMemoryStore()
    await store._ensure_collection()
    await store._ensure_collection()
    mock_client.collection_exists.assert_called_once()


async def test_init_custom_params(mock_client: MagicMock) -> None:
    store = QdrantMemoryStore(
        collection_name="my_eps",
        embedding_model="BAAI/bge-large-en-v1.5",
    )
    mock_client.set_model.assert_called_once_with("BAAI/bge-large-en-v1.5")
    assert store._collection_name == "my_eps"


def test_init_custom_client() -> None:
    custom_client = MagicMock(spec=AsyncQdrantClient)
    store = QdrantMemoryStore(client=custom_client)
    assert store._client is custom_client
    custom_client.set_model.assert_called_once()


async def test_write(mock_client: MagicMock, episode: Episode) -> None:
    store = QdrantMemoryStore()
    await store.write(episode)

    mock_client.upsert.assert_called_once()
    kw = mock_client.upsert.call_args.kwargs
    assert kw["collection_name"] == "episodes"
    point = kw["points"][0]
    assert point.id == episode.id_
    assert (
        episode.task.instruction in point.vector["fast-bge-small-en-v1.5"].text
    )
    assert episode.result.content in point.vector["fast-bge-small-en-v1.5"].text
    assert "episode_json" in point.payload
    assert "completed_at" in point.payload


async def test_write_uses_key_fn_when_provided(
    mock_client: MagicMock,
    episode: Episode,
) -> None:
    custom_text = "custom formatted text"
    store = QdrantMemoryStore(key_fn=lambda ep: custom_text)
    await store.write(episode)

    kw = mock_client.upsert.call_args.kwargs
    point = kw["points"][0]
    assert point.vector["fast-bge-small-en-v1.5"].text == custom_text


async def test_count(mock_client: MagicMock) -> None:
    expected = 7
    mock_client.count.return_value = MagicMock(count=expected)
    store = QdrantMemoryStore()
    assert await store.count() == expected


async def test_read_recent_empty(mock_client: MagicMock) -> None:
    mock_client.scroll.return_value = ([], None)
    store = QdrantMemoryStore()
    assert await store._read_recent(3) == []


async def test_read_recent(mock_client: MagicMock, episode: Episode) -> None:
    record = MagicMock()
    record.payload = {
        "episode_json": episode.model_dump_json(),
        "completed_at": episode.completed_at.timestamp(),
    }
    mock_client.scroll.return_value = ([record], None)

    store = QdrantMemoryStore()
    result = await store._read_recent(5)

    assert len(result) == 1
    assert result[0].task.instruction == episode.task.instruction


async def test_read_recent_passes_order_by_to_scroll(
    mock_client: MagicMock,
) -> None:
    mock_client.scroll.return_value = ([], None)
    store = QdrantMemoryStore()
    await store._read_recent(2)

    kw = mock_client.scroll.call_args.kwargs
    assert kw["order_by"].key == "completed_at"
    assert kw["order_by"].direction == models.Direction.DESC


async def test_read_recent_passes_n_as_limit_to_scroll(
    mock_client: MagicMock,
) -> None:
    episodes = []
    for i in range(3):
        t = Task(instruction=f"task {i}")
        episodes.append(
            Episode(
                task=t,
                rollout="",
                result=TaskResult(task_id=t.id_, content=f"result {i}"),
                completed_at=datetime(2025, 1, i + 1),
            ),
        )

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
    result = await store._read_recent(n)

    assert mock_client.scroll.call_args.kwargs["limit"] == n
    assert len(result) == n


async def test_search(mock_client: MagicMock, episode: Episode) -> None:
    mock_point = MagicMock()
    mock_point.payload = {"episode_json": episode.model_dump_json()}
    mock_client.query_points.return_value = MagicMock(points=[mock_point])

    max_results = 3
    store = QdrantMemoryStore(max_results=max_results)
    results = await store.search("electric type pokemon")

    mock_client.query_points.assert_called_once()
    kw = mock_client.query_points.call_args.kwargs
    assert kw["collection_name"] == "episodes"
    assert kw["query"].text == "electric type pokemon"
    assert kw["using"] == "fast-bge-small-en-v1.5"
    assert kw["limit"] == max_results
    assert len(results) == 1
    assert results[0].task.instruction == episode.task.instruction


async def test_search_empty(mock_client: MagicMock) -> None:
    mock_client.query_points.return_value = MagicMock(points=[])
    store = QdrantMemoryStore()
    results = await store.search("anything")
    assert results == []


async def test_search_uses_read_recent_when_recall_mode_recent(
    mock_client: MagicMock,
    episode: Episode,
) -> None:
    mock_client.count.return_value = MagicMock(count=1)
    mock_point = MagicMock()
    mock_point.payload = {
        "episode_json": episode.model_dump_json(),
        "completed_at": episode.completed_at.timestamp(),
    }
    mock_client.scroll.return_value = ([mock_point], None)

    store = QdrantMemoryStore(recall_mode=RecallMode.RECENT, max_results=5)
    results = await store.search("ignored query")

    mock_client.query_points.assert_not_called()
    assert len(results) == 1
    assert results[0].id_ == episode.id_


async def test_summary_empty(mock_client: MagicMock) -> None:
    mock_client.count.return_value = MagicMock(count=0)
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

    mock_client.count.return_value = MagicMock(count=1)
    mock_client.scroll.return_value = ([record], None)

    store = QdrantMemoryStore()
    summary = await store.summary()

    assert "QdrantMemoryStore" in summary
    assert "1" in summary
    assert episode.task.instruction[:20] in summary


# --- delete ---


@pytest.mark.asyncio
async def test_delete_removes_existing_episode(
    mock_client: MagicMock,
) -> None:
    store = QdrantMemoryStore()
    hit = MagicMock()
    hit.id = "some-id"
    mock_client.retrieve.return_value = [hit]

    await store.delete("some-id")

    mock_client.delete.assert_called_once_with(
        collection_name=store._collection_name,
        points_selector=models.PointIdsList(points=["some-id"]),
    )


@pytest.mark.asyncio
async def test_delete_warns_when_id_not_found(
    mock_client: MagicMock,
) -> None:
    store = QdrantMemoryStore()
    mock_client.retrieve.return_value = []

    with pytest.raises(EpisodeNotFoundError):
        await store.delete("nonexistent-id")

    mock_client.delete.assert_not_called()


# --- update ---


@pytest.mark.asyncio
async def test_update_upserts_existing_episode(
    mock_client: MagicMock,
    episode: Episode,
) -> None:
    store = QdrantMemoryStore()
    hit = MagicMock()
    hit.id = episode.id_
    mock_client.retrieve.return_value = [hit]

    await store.update(episode)

    mock_client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_update_warns_when_id_not_found(
    mock_client: MagicMock,
    episode: Episode,
) -> None:
    store = QdrantMemoryStore()
    mock_client.retrieve.return_value = []

    with pytest.raises(EpisodeNotFoundError):
        await store.update(episode)

    mock_client.upsert.assert_not_called()
