"""Qdrant-backed episodic memory store."""

from typing import Any

from qdrant_client import QdrantClient

from llm_agents_from_scratch.base.memory import BaseMemoryStore
from llm_agents_from_scratch.data_structures.memory import Episode


class QdrantMemoryStore(BaseMemoryStore):
    """Episodic memory store backed by a Qdrant collection.

    Episodes are embedded at write time using FastEmbed and stored as
    vector points. Similarity search uses cosine distance over the
    embedded episode text (task instruction concatenated with result
    content).

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
        """
        self._client = client or QdrantClient(":memory:")
        self._client.set_model(embedding_model)
        self._collection = collection_name
        if not self._client.collection_exists(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=self._client.get_fastembed_vector_params(),
            )

    async def write(self, episode: Episode) -> None:
        """Embed and persist an episode to the Qdrant collection.

        The embedded text is the episode's task instruction followed by
        the result content, separated by a newline. The full serialised
        episode and its completion timestamp are stored in the point
        payload for later retrieval.

        Args:
            episode (Episode): The completed episode to store.
        """
        text = f"{episode.task.instruction}\n{episode.result.content}"
        self._client.add(
            collection_name=self._collection,
            documents=[text],
            metadata=[
                {
                    "episode_json": episode.model_dump_json(),
                    "completed_at": episode.completed_at.timestamp(),
                },
            ],
            ids=[episode.task.id_],
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
        episodes = [
            Episode.model_validate_json(p.payload["episode_json"])
            for p in points
            if p.payload
        ]
        return sorted(
            episodes,
            key=lambda e: e.completed_at,
            reverse=True,
        )[:n]

    async def count(self) -> int:
        """Return the total number of episodes in the store.

        Returns:
            int: Episode count.
        """
        return int(self._client.count(self._collection).count)

    async def search(
        self,
        query: str,
        k: int,
        **kwargs: Any,
    ) -> list[Episode]:
        """Return the K episodes most semantically similar to a query.

        Embeds the query using the same FastEmbed model used at write
        time and retrieves the top-K points by cosine similarity.

        Args:
            query (str): The search query (e.g. the task instruction).
            k (int): Maximum number of episodes to return.
            **kwargs: Additional keyword arguments forwarded to
                ``QdrantClient.query()`` (e.g. ``query_filter``,
                ``score_threshold``).

        Returns:
            list[Episode]: Episodes ordered by cosine similarity to the
                query.
        """
        results = self._client.query(
            collection_name=self._collection,
            query_text=query,
            limit=k,
            **kwargs,
        )
        return [
            Episode.model_validate_json(r.metadata["episode_json"])
            for r in results
            if r.metadata
        ]
