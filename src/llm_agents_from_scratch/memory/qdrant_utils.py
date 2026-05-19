"""Qdrant data-conversion utilities for episodic memory."""

from qdrant_client import models

from llm_agents_from_scratch.data_structures.memory import Episode


def episode_to_qdrant_point_struct(
    episode: Episode,
    vector_field: str,
    model_name: str,
) -> models.PointStruct:
    """Convert an episode to a Qdrant PointStruct ready for upsert.

    The embedded text is the episode's task instruction followed by the
    result content, separated by a newline. XML tags and timestamps from
    ``Episode.__str__`` are intentionally excluded — they add noise and
    allow recency to bleed into relevance scores.

    Args:
        episode (Episode): The completed episode to convert.
        vector_field (str): Name of the vector field in the Qdrant
            collection (e.g. ``"fast-bge-small-en-v1.5"``).
        model_name (str): FastEmbed model identifier used for embedding
            (e.g. ``"BAAI/bge-small-en-v1.5"``).

    Returns:
        models.PointStruct: A point ready to pass to
            ``QdrantClient.upsert()``.
    """
    text = episode.format(mode="concat")
    return models.PointStruct(
        id=episode.task.id_,
        vector={
            vector_field: models.Document(text=text, model=model_name),
        },
        payload={
            "episode_json": episode.model_dump_json(),
            "completed_at": episode.completed_at.timestamp(),
        },
    )
