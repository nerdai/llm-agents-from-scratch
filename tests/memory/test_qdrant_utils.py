import pytest
from qdrant_client import models

from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.memory import (
    Episode,
    EpisodeFormatMode,
)
from llm_agents_from_scratch.memory_stores.qdrant.errors import (
    QdrantEpisodeJsonMissingError,
    QdrantPointPayloadMissingError,
)
from llm_agents_from_scratch.memory_stores.qdrant.utils import (
    episode_to_qdrant_point_struct,
    qdrant_point_to_episode,
)

_FIELD = "fast-bge-small-en-v1.5"
_MODEL = "BAAI/bge-small-en-v1.5"


def _make_episode(instruction: str = "look up pikachu") -> Episode:
    task = Task(instruction=instruction)
    return Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="pikachu is electric"),
    )


def test_id_matches_task_id() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        "some text",
        vector_field=_FIELD,
        model_name=_MODEL,
    )
    assert point.id == episode.id_


def test_vector_field_name() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        "some text",
        vector_field=_FIELD,
        model_name=_MODEL,
    )
    assert _FIELD in point.vector  # type: ignore[operator]


def test_document_model_name() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        "some text",
        vector_field=_FIELD,
        model_name=_MODEL,
    )
    assert point.vector[_FIELD].model == _MODEL  # type: ignore[index]


def test_text_passed_through_to_document() -> None:
    episode = _make_episode()
    text = episode.format(
        mode=EpisodeFormatMode.CONCAT,
        exclude={"id_", "rollout", "error", "metadata", "completed_at"},
    )
    point = episode_to_qdrant_point_struct(
        episode,
        text,
        vector_field=_FIELD,
        model_name=_MODEL,
    )
    doc = point.vector[_FIELD]  # type: ignore[index]
    assert episode.task.instruction in doc.text  # type: ignore[union-attr]
    assert episode.result.content in doc.text  # type: ignore[union-attr]


def test_metadata_in_text_when_caller_includes_it() -> None:
    task = Task(instruction="look up pikachu")
    episode = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="pikachu is electric"),
        metadata={"reflection": "Always verify the name spelling."},
    )
    text = episode.format(
        mode=EpisodeFormatMode.CONCAT,
        exclude={"id_", "rollout", "error", "completed_at"},
    )
    point = episode_to_qdrant_point_struct(
        episode,
        text,
        vector_field=_FIELD,
        model_name=_MODEL,
    )
    doc = point.vector[_FIELD]  # type: ignore[index]
    assert "Always verify the name spelling." in doc.text  # type: ignore[union-attr]


def test_payload_contains_episode_json() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        "some text",
        vector_field=_FIELD,
        model_name=_MODEL,
    )
    assert "episode_json" in point.payload
    assert "completed_at" in point.payload


def test_payload_episode_json_roundtrips() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        "some text",
        vector_field=_FIELD,
        model_name=_MODEL,
    )
    restored = Episode.model_validate_json(point.payload["episode_json"])
    assert restored.task.instruction == episode.task.instruction
    assert restored.result.content == episode.result.content


def test_custom_vector_field_and_model() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        "some text",
        vector_field="my-custom-field",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    assert "my-custom-field" in point.vector  # type: ignore[operator]
    assert point.vector["my-custom-field"].model == (  # type: ignore[index]
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# --- qdrant_point_to_episode ---


def test_qdrant_point_to_episode_from_record() -> None:
    episode = _make_episode()
    record = models.Record(
        id=episode.id_,
        payload={"episode_json": episode.model_dump_json()},
    )
    restored = qdrant_point_to_episode(record)
    assert restored.id_ == episode.id_
    assert restored.task.instruction == episode.task.instruction
    assert restored.result.content == episode.result.content


def test_qdrant_point_to_episode_from_scored_point() -> None:
    episode = _make_episode()
    scored = models.ScoredPoint(
        id=episode.id_,
        version=0,
        score=0.95,
        payload={"episode_json": episode.model_dump_json()},
        vector=None,
    )
    restored = qdrant_point_to_episode(scored)
    assert restored.id_ == episode.id_
    assert restored.task.instruction == episode.task.instruction


def test_qdrant_point_to_episode_roundtrip() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        "text",
        vector_field=_FIELD,
        model_name=_MODEL,
    )
    record = models.Record(id=point.id, payload=point.payload)
    restored = qdrant_point_to_episode(record)
    assert restored == episode


def test_qdrant_point_to_episode_raises_when_payload_none() -> None:
    record = models.Record(id="some-id", payload=None)
    with pytest.raises(QdrantPointPayloadMissingError, match="some-id"):
        qdrant_point_to_episode(record)


def test_qdrant_point_to_episode_raises_when_episode_json_missing() -> None:
    record = models.Record(id="some-id", payload={"completed_at": 1234567890})
    with pytest.raises(QdrantEpisodeJsonMissingError, match="some-id"):
        qdrant_point_to_episode(record)


def test_qdrant_point_to_episode_empty_payload_raises() -> None:
    record = models.Record(id="some-id", payload={})
    with pytest.raises(QdrantEpisodeJsonMissingError, match="some-id"):
        qdrant_point_to_episode(record)
