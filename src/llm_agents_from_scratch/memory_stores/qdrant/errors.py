"""Errors for Qdrant memory store integration."""

from llm_agents_from_scratch.errors.core import LLMAgentsFromScratchError


class QdrantMemoryStoreError(LLMAgentsFromScratchError):
    """Base error for all Qdrant memory store exceptions."""

    pass


class QdrantPointPayloadError(QdrantMemoryStoreError):
    """Base error for Qdrant point payload problems."""

    pass


class QdrantPointPayloadMissingError(QdrantPointPayloadError):
    """Raised when a Qdrant point has no payload (payload is None)."""

    pass


class QdrantEpisodeJsonMissingError(QdrantPointPayloadError):
    """Raised when a point payload exists but lacks the episode_json key."""

    pass


__all__ = [
    "QdrantMemoryStoreError",
    "QdrantPointPayloadError",
    "QdrantPointPayloadMissingError",
    "QdrantEpisodeJsonMissingError",
]
