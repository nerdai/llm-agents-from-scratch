"""Qdrant data-conversion utilities for episodic memory."""

from qdrant_client import models

from llm_agents_from_scratch.data_structures.memory import Episode


def episode_to_qdrant_point_struct(
    episode: Episode,
    text: str,
    vector_field: str,
    model_name: str,
) -> models.PointStruct:
    """Convert an episode to a Qdrant PointStruct ready for upsert.

    The caller is responsible for formatting ``text`` — typically via
    ``Episode.format()`` — so that memory strategies own the embedding
    content decision.

    Args:
        episode (Episode): The completed episode to convert.
        text (str): Pre-formatted text to embed as the point vector.
        vector_field (str): Name of the vector field in the Qdrant
            collection (e.g. ``"fast-bge-small-en-v1.5"``).
        model_name (str): FastEmbed model identifier used for embedding
            (e.g. ``"BAAI/bge-small-en-v1.5"``).

    Returns:
        models.PointStruct: A point ready to pass to
            ``AsyncQdrantClient.upsert()``.
    """
    return models.PointStruct(
        id=episode.id_,
        vector={
            vector_field: models.Document(text=text, model=model_name),
        },
        payload={
            "episode_json": episode.model_dump_json(),
            "completed_at": episode.completed_at.timestamp(),
        },
    )


def qdrant_point_to_episode(
    point: models.Record | models.ScoredPoint,
) -> Episode:
    """Convert a Qdrant read point back to an Episode.

    Inverse of ``episode_to_qdrant_point_struct``. Accepts both
    ``models.Record`` (returned by ``scroll`` / ``retrieve``) and
    ``models.ScoredPoint`` (returned by ``query_points``).

    Args:
        point (models.Record | models.ScoredPoint): A point retrieved
            from a Qdrant collection.

    Returns:
        Episode: The deserialized episode.

    Raises:
        KeyError: If the point payload does not contain ``episode_json``.
        ValueError: If ``episode_json`` cannot be parsed as an
            ``Episode``.
    """
    return Episode.model_validate_json(point.payload["episode_json"])  # type: ignore[index]
