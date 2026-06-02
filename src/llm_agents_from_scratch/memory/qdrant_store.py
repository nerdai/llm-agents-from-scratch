"""Qdrant-backed episodic memory store."""

from typing import Any

from qdrant_client import QdrantClient, models

from llm_agents_from_scratch.base.memory import BaseMemoryStore
from llm_agents_from_scratch.data_structures.memory import Episode, EpisodeAttr
from llm_agents_from_scratch.memory.qdrant_utils import (
    episode_to_qdrant_point_struct,
)

DEFAULT_EPISODE_INCLUDE: list[EpisodeAttr] = [
    "instruction",
    "result",
    "metadata",
]


class QdrantMemoryStore(BaseMemoryStore):
    """Episodic memory store backed by a Qdrant collection.

    Episodes are embedded at write time using FastEmbed and stored as
    vector points. Similarity search uses cosine distance over the
    embedded episode text.

    By default, Qdrant runs in-process with no server required. Pass a
    pre-configured ``QdrantClient`` to persist to a remote or on-disk
    instance instead.

    Note: This is a minimal integration. Advanced Qdrant features such
    as payload filters, score thresholds, and named vectors are not
    exposed. Use the ``_client`` attribute directly for full API access.

    Attributes:
        _client (QdrantClient): The Qdrant client instance.
        _collection (str): Name of the Qdrant collection.
    """

    def __init__(
        self,
        collection_name: str = "episodes",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        client: QdrantClient | None = None,
        max_results: int = 5,
    ) -> None:
        """Initialize a QdrantMemoryStore.

        Creates a Qdrant collection configured for cosine similarity over
        FastEmbed vectors. Embedding runs locally via ONNX Runtime; no
        external embedding service is required. The model is downloaded
        on first use and cached locally.

        Args:
            collection_name (str): Name of the Qdrant collection.
                Defaults to ``"episodes"``.
            embedding_model (str): FastEmbed model name used to embed
                episode text at write time and queries at search time.
                Defaults to ``"BAAI/bge-small-en-v1.5"``.
            client (QdrantClient | None): Pre-configured Qdrant client.
                Defaults to an in-memory client when ``None``. The
                client must use FastEmbed as its embedding backend.
            max_results (int): Default maximum number of episodes
                returned by ``search``. Defaults to 5.
        """
        super().__init__(max_results=max_results)
        self._client = client or QdrantClient(":memory:")
        self._client.set_model(embedding_model)
        self._collection = collection_name
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=self._client.get_fastembed_vector_params(),
            )

    async def write(
        self,
        episode: Episode,
        key: str | None = None,
    ) -> None:
        """Embed and persist an episode to the Qdrant collection.

        The full serialised episode and its completion timestamp are
        stored in the point payload for later retrieval. The key
        defaults to a concat-format serialisation of
        ``DEFAULT_EPISODE_INCLUDE`` attributes when ``key`` is not
        provided.

        Args:
            episode (Episode): The completed episode to store.
            key (str | None): Pre-formatted text to embed. When
                provided by the calling memory strategy, this text is
                used directly for the vector. Defaults to ``None``, in
                which case the store formats the episode using
                ``DEFAULT_EPISODE_INCLUDE``.
        """
        text = key or episode.format(
            mode="concat",
            include=DEFAULT_EPISODE_INCLUDE,
        )
        self._client.upsert(
            collection_name=self._collection,
            points=[
                episode_to_qdrant_point_struct(
                    episode,
                    text,
                    self._client.get_vector_field_name(),
                    self._client.embedding_model_name,
                ),
            ],
        )

    async def read_recent(self, n: int) -> list[Episode]:
        """Return the N most recently recorded episodes.

        Fetches all points from the collection and sorts by the stored
        ``completed_at`` timestamp.

        Args:
            n (int): Maximum number of episodes to return.

        Returns:
            list[Episode]: Episodes ordered from most recent to oldest.
        """
        total = int(self._client.count(self._collection).count)
        if total == 0:
            return []
        points, _ = self._client.scroll(
            collection_name=self._collection,
            with_payload=True,
            limit=total,
        )
        valid = [
            p
            for p in points
            if p.payload
            and "episode_json" in p.payload
            and "completed_at" in p.payload
        ]
        valid.sort(
            key=lambda p: p.payload["completed_at"],  # type: ignore[index]
            reverse=True,
        )
        return [
            Episode.model_validate_json(p.payload["episode_json"])  # type: ignore[index]
            for p in valid[:n]
        ]

    async def count(self) -> int:
        """Return the total number of episodes in the store.

        Returns:
            int: Episode count.
        """
        return int(self._client.count(self._collection).count)

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
            f" | collection={self._collection}",
        ]
        if total > 0:
            episodes = await self.read_recent(total)
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

    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[Episode]:
        """Return the most semantically similar episodes for a query.

        Embeds the query using the same FastEmbed model used at write
        time and retrieves the top ``max_results`` points by cosine
        similarity.

        Args:
            query (str): The search query (e.g. the task instruction).
            **kwargs: Additional keyword arguments forwarded to
                ``QdrantClient.query_points()`` (e.g. ``query_filter``,
                ``score_threshold``).

        Returns:
            list[Episode]: Episodes ordered by cosine similarity to the
                query.
        """
        results = self._client.query_points(
            collection_name=self._collection,
            query=models.Document(
                text=query,
                model=self._client.embedding_model_name,
            ),
            using=self._client.get_vector_field_name(),
            limit=self.max_results,
            with_payload=True,
            **kwargs,
        ).points
        return [
            Episode.model_validate_json(r.payload["episode_json"])
            for r in results
            if r.payload and "episode_json" in r.payload
        ]
