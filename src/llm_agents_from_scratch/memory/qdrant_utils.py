"""Qdrant data-conversion utilities for episodic memory."""

from typing import Literal

from qdrant_client import models

from llm_agents_from_scratch.data_structures.memory import Episode, EpisodeAttr


def episode_to_qdrant_point_struct(
    episode: Episode,
    mode: Literal["xml", "concat"] = "concat",
    include: list[EpisodeAttr] | None = None,
    # qdrant
    *,
    vector_field: str,
    model_name: str,
) -> models.PointStruct:
    """Convert an episode to a Qdrant PointStruct ready for upsert.

    Args:
        episode (Episode): The completed episode to convert.
        mode (Literal["xml", "concat"]): Episode serialisation mode
            passed to ``Episode.format()``. Defaults to ``"concat"``.
        include (list[EpisodeAttr] | None): Attributes to include in
            the embedded text. Defaults to ``Episode.format()``
            defaults for the given mode.
        vector_field (str): Name of the vector field in the Qdrant
            collection (e.g. ``"fast-bge-small-en-v1.5"``).
        model_name (str): FastEmbed model identifier used for embedding
            (e.g. ``"BAAI/bge-small-en-v1.5"``).


    Returns:
        models.PointStruct: A point ready to pass to
            ``QdrantClient.upsert()``.
    """
    text = episode.format(mode=mode, include=include)
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
