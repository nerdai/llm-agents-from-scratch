"""Qdrant-backed episodic memory store."""

from typing import Any, Callable

from qdrant_client import AsyncQdrantClient, models

from llm_agents_from_scratch.base.memory_store import BaseMemoryStore
from llm_agents_from_scratch.data_structures.memory import (
    Episode,
    EpisodeFormatMode,
    RecallMode,
)
from llm_agents_from_scratch.errors import EpisodeNotFoundError
from llm_agents_from_scratch.memory_stores.qdrant.utils import (
    episode_to_qdrant_point_struct,
    qdrant_point_to_episode,
)

DEFAULT_EPISODE_EXCLUDE: set[str] = {"id_", "rollout", "completed_at"}


class QdrantMemoryStore(BaseMemoryStore):
    """Episodic memory store backed by a Qdrant collection.

    Episodes are embedded at write time using FastEmbed and stored as
    vector points. Similarity search uses cosine distance over the
    embedded episode text.

    By default, Qdrant runs in-process with no server required. Pass a
    pre-configured ``AsyncQdrantClient`` to persist to a remote or on-disk
    instance instead.

    Note: This is a minimal integration. Advanced Qdrant features such
    as payload filters, score thresholds, and named vectors are not
    exposed. Use the ``_client`` attribute directly for full API access.

    Attributes:
        _client (AsyncQdrantClient): The Qdrant client instance.
        _collection_name (str): Name of the Qdrant collection.
        _key_fn (Callable[[Episode], str]): Callable that extracts the
            text used for embedding from an episode.
    """

    def __init__(  # noqa: PLR0913
        self,
        collection_name: str = "episodes",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        client: AsyncQdrantClient | None = None,
        max_results: int = 5,
        recall_mode: RecallMode = RecallMode.SEARCH,
        key_fn: Callable[[Episode], str] | None = None,
    ) -> None:
        """Initialize a QdrantMemoryStore.

        The Qdrant collection is created lazily on the first operation
        via ``_ensure_collection()``. No async calls are made in
        ``__init__``.

        Args:
            collection_name (str): Name of the Qdrant collection.
                Defaults to ``"episodes"``.
            embedding_model (str): FastEmbed model name used to embed
                episode text at write time and queries at search time.
                Defaults to ``"BAAI/bge-small-en-v1.5"``.
            client (AsyncQdrantClient | None): Pre-configured async
                Qdrant client. Defaults to an in-memory client when
                ``None``. The client must use FastEmbed as its embedding
                backend.
            max_results (int): Default maximum number of episodes
                returned by ``search``. Defaults to 5.
            recall_mode (RecallMode): Retrieval strategy used by
                ``search()``. Defaults to ``RecallMode.SEARCH``.
            key_fn (Callable[[Episode], str] | None): Callable that
                extracts the text to embed from an episode. Defaults to
                a concat-format serialization using
                ``DEFAULT_EPISODE_EXCLUDE``.
        """
        super().__init__(max_results=max_results, recall_mode=recall_mode)
        self._client = client or AsyncQdrantClient(":memory:")
        self._client.set_model(embedding_model)
        self._collection_name = collection_name
        self._collection_ready = False
        self._key_fn: Callable[[Episode], str] = key_fn or (
            lambda ep: ep.format(
                mode=EpisodeFormatMode.CONCAT,
                exclude=DEFAULT_EPISODE_EXCLUDE,
            )
        )

    async def _ensure_collection(self) -> None:
        """Create the Qdrant collection and payload index if they do not exist.

        Creates a float payload index on ``completed_at`` so that
        ``order_by`` in ``scroll()`` works against a real Qdrant server
        (in-memory mode does not require an index, but server mode does).

        No-op after the first successful call (guarded by
        ``_collection_ready``).
        """
        if self._collection_ready:
            return
        if not await self._client.collection_exists(self._collection_name):
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=self._client.get_fastembed_vector_params(),
            )
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="completed_at",
                field_schema=models.PayloadSchemaType.FLOAT,
            )
        self._collection_ready = True

    async def write(self, episode: Episode) -> None:
        """Embed and persist an episode to the Qdrant collection.

        The full serialized episode and its completion timestamp are
        stored in the point payload for later retrieval. The embedding
        text is produced by ``_key_fn``.

        Args:
            episode (Episode): The completed episode to store.
        """
        await self._ensure_collection()
        text = self._key_fn(episode)
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                episode_to_qdrant_point_struct(
                    episode,
                    text,
                    self._client.get_vector_field_name(),
                    self._client.embedding_model_name,
                ),
            ],
        )

    async def _read_recent(self, n: int) -> list[Episode]:
        """Return the N most recently recorded episodes.

        Uses Qdrant's ``order_by`` to sort by ``completed_at`` server-side
        so only the top ``n`` points are transferred.

        Args:
            n (int): Maximum number of episodes to return.

        Returns:
            list[Episode]: Episodes ordered from most recent to oldest.
        """
        await self._ensure_collection()
        points, _ = await self._client.scroll(
            collection_name=self._collection_name,
            with_payload=True,
            limit=n,
            order_by=models.OrderBy(
                key="completed_at",
                direction=models.Direction.DESC,
            ),
        )
        return [qdrant_point_to_episode(p) for p in points]

    async def count(self) -> int:
        """Return the total number of episodes in the store.

        Returns:
            int: Episode count.
        """
        await self._ensure_collection()
        return int((await self._client.count(self._collection_name)).count)

    async def _point_exists(self, id_: str) -> bool:
        """Return True if a point with ``id_`` exists in the collection."""
        hits = await self._client.retrieve(
            collection_name=self._collection_name,
            ids=[id_],
        )
        return id_ in {h.id for h in hits}

    async def delete(self, id_: str) -> None:
        """Delete an episode by its unique identifier.

        Raises ``EpisodeNotFoundError`` if no point with ``id_`` exists
        in the collection. Otherwise removes it via
        ``AsyncQdrantClient.delete``.

        Args:
            id_ (str): The ``Episode.id_`` of the episode to remove.

        Raises:
            EpisodeNotFoundError: If no point with ``id_`` exists.
        """
        await self._ensure_collection()
        if not await self._point_exists(id_):
            raise EpisodeNotFoundError(
                f"Episode '{id_}' not found in QdrantMemoryStore.",
            )
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=models.PointIdsList(points=[id_]),
        )

    async def update(self, episode: Episode) -> None:
        """Replace an existing episode with an updated version.

        Matches by ``episode.id_`` (the Qdrant point ID). Raises
        ``EpisodeNotFoundError`` if no matching point exists. Otherwise
        re-embeds and upserts the updated episode using ``_key_fn``.

        Args:
            episode (Episode): The updated episode. Matched by ``id_``.

        Raises:
            EpisodeNotFoundError: If no point with ``episode.id_`` exists.
        """
        await self._ensure_collection()
        if not await self._point_exists(episode.id_):
            raise EpisodeNotFoundError(
                f"Episode '{episode.id_}' not found in QdrantMemoryStore.",
            )
        text = self._key_fn(episode)
        await self._client.upsert(
            collection_name=self._collection_name,
            points=[
                episode_to_qdrant_point_struct(
                    episode,
                    text,
                    self._client.get_vector_field_name(),
                    self._client.embedding_model_name,
                ),
            ],
        )

    async def summary(self) -> str:
        """Return a human-readable summary of the store contents.

        Includes the collection name, total episode count, and the
        instruction and timestamp of the newest and oldest episodes.

        Returns:
            str: Multi-line summary of the store.
        """
        total = await self.count()
        lines = [
            f"QdrantMemoryStore: {total} episodes"
            f" | collection={self._collection_name}",
        ]
        if total > 0:
            episodes = await self._read_recent(total)
            newest = episodes[0]
            oldest = episodes[-1]
            lines.append(
                f"  newest: {str(newest.completed_at)[:19]}"
                f" | {newest.task.instruction[:60]}",
            )
            lines.append(
                f"  oldest: {str(oldest.completed_at)[:19]}"
                f" | {oldest.task.instruction[:60]}",
            )
        return "\n".join(lines)

    async def _search(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[Episode]:
        """Return the top ``max_results`` episodes most similar to ``query``.

        Called by the base ``search()`` when ``recall_mode`` is
        ``RecallMode.SEARCH``. Uses cosine similarity over FastEmbed vectors.

        Args:
            query (str): The search query (e.g. the task instruction).
            **kwargs: Additional keyword arguments forwarded to
                ``AsyncQdrantClient.query_points()``
                (e.g. ``query_filter``, ``score_threshold``).

        Returns:
            list[Episode]: Episodes ordered by cosine similarity.
        """
        await self._ensure_collection()
        similar_points = (
            await self._client.query_points(
                collection_name=self._collection_name,
                query=models.Document(
                    text=query,
                    model=self._client.embedding_model_name,
                ),
                using=self._client.get_vector_field_name(),
                limit=self.max_results,
                with_payload=True,
                **kwargs,
            )
        ).points
        return [qdrant_point_to_episode(p) for p in similar_points]
