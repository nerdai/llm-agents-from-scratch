from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.memory import Episode
from llm_agents_from_scratch.memory.qdrant_utils import (
    episode_to_qdrant_point_struct,
)


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
        vector_field="fast-bge-small-en-v1.5",
        model_name="BAAI/bge-small-en-v1.5",
    )
    assert point.id == episode.task.id_


def test_vector_field_name() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        vector_field="fast-bge-small-en-v1.5",
        model_name="BAAI/bge-small-en-v1.5",
    )
    assert "fast-bge-small-en-v1.5" in point.vector


def test_document_model_name() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        vector_field="fast-bge-small-en-v1.5",
        model_name="BAAI/bge-small-en-v1.5",
    )
    doc = point.vector["fast-bge-small-en-v1.5"]
    assert doc.model == "BAAI/bge-small-en-v1.5"


def test_embedded_text_contains_instruction_and_result() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        vector_field="fast-bge-small-en-v1.5",
        model_name="BAAI/bge-small-en-v1.5",
    )
    doc = point.vector["fast-bge-small-en-v1.5"]
    assert episode.task.instruction in doc.text
    assert episode.result.content in doc.text


def test_embedded_text_includes_additional_data() -> None:
    task = Task(instruction="look up pikachu")
    episode = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="pikachu is electric"),
        additional_data={"reflection": "Always verify the name spelling."},
    )
    point = episode_to_qdrant_point_struct(
        episode,
        vector_field="fast-bge-small-en-v1.5",
        model_name="BAAI/bge-small-en-v1.5",
    )
    doc = point.vector["fast-bge-small-en-v1.5"]  # type: ignore[index]
    assert "Always verify the name spelling." in doc.text  # type: ignore[union-attr]


def test_payload_contains_episode_json() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        vector_field="fast-bge-small-en-v1.5",
        model_name="BAAI/bge-small-en-v1.5",
    )
    assert "episode_json" in point.payload
    assert "completed_at" in point.payload


def test_payload_episode_json_roundtrips() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        vector_field="fast-bge-small-en-v1.5",
        model_name="BAAI/bge-small-en-v1.5",
    )
    restored = Episode.model_validate_json(point.payload["episode_json"])
    assert restored.task.instruction == episode.task.instruction
    assert restored.result.content == episode.result.content


def test_custom_vector_field_and_model() -> None:
    episode = _make_episode()
    point = episode_to_qdrant_point_struct(
        episode,
        vector_field="my-custom-field",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    assert "my-custom-field" in point.vector
    assert point.vector["my-custom-field"].model == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
