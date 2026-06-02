"""Unit tests for SimilarityMemory recipe."""

from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.memory import Episode, RecallMode
from llm_agents_from_scratch.memory import Memory, SimilarityMemory
from llm_agents_from_scratch.memory_stores.qdrant.store import QdrantMemoryStore


def make_episode(
    instruction: str = "do something",
    content: str = "done",
) -> Episode:
    task = Task(instruction=instruction)
    return Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content=content),
    )


def test_returns_memory_instance() -> None:
    memory = SimilarityMemory()
    assert isinstance(memory, Memory)


def test_store_is_qdrant() -> None:
    memory = SimilarityMemory()
    assert isinstance(memory.store, QdrantMemoryStore)


def test_store_recall_mode_is_search() -> None:
    memory = SimilarityMemory()
    assert memory.store.recall_mode == RecallMode.SEARCH


def test_max_results_set_from_k() -> None:
    memory = SimilarityMemory(k=7)
    assert memory.store.max_results == 7  # noqa: PLR2004


def test_default_key_fn_uses_instruction() -> None:
    memory = SimilarityMemory()
    ep = make_episode("look up pikachu")
    assert memory.key_fn(ep) == "look up pikachu"


def test_no_metadata_fns_by_default() -> None:
    memory = SimilarityMemory()
    assert memory.metadata_fns == {}
