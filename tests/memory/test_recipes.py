"""Unit tests for memory factory functions in memory/recipes.py."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.llm import CompleteResult
from llm_agents_from_scratch.data_structures.memory import Episode
from llm_agents_from_scratch.memory import Memory
from llm_agents_from_scratch.memory.recipes import (
    recency_memory,
    reflective_memory,
    similarity_memory,
)
from llm_agents_from_scratch.memory_stores.json import JSONMemoryStore
from llm_agents_from_scratch.memory_stores.qdrant.store import QdrantMemoryStore

_QDRANT_PATCH = (
    "llm_agents_from_scratch.memory_stores.qdrant.store.QdrantClient"
)


@pytest.fixture
def mock_qdrant_client():
    with patch(_QDRANT_PATCH) as mock_cls:
        instance = MagicMock()
        instance.embedding_model_name = "BAAI/bge-small-en-v1.5"
        instance.get_vector_field_name.return_value = "fast-bge-small-en-v1.5"
        mock_cls.return_value = instance
        yield instance


def make_episode() -> Episode:
    task = Task(instruction="test task")
    return Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="test result"),
    )


def make_llm(reflection: str = "use more samples") -> BaseLLM:
    llm = MagicMock(spec=BaseLLM)
    llm.complete = AsyncMock(
        return_value=CompleteResult(response=reflection, prompt=""),
    )
    return llm


# --- recency_memory ---


def test_recency_memory_returns_memory(tmp_path: Path) -> None:
    memory = recency_memory(path=tmp_path)
    assert isinstance(memory, Memory)


def test_recency_memory_store_is_json(tmp_path: Path) -> None:
    memory = recency_memory(path=tmp_path)
    assert isinstance(memory.store, JSONMemoryStore)


def test_recency_memory_max_results(tmp_path: Path) -> None:
    memory = recency_memory(path=tmp_path, max_results=3)
    assert memory.store.max_results == 3  # noqa: PLR2004


# --- similarity_memory ---


def test_similarity_memory_returns_memory(
    mock_qdrant_client: MagicMock,
) -> None:
    memory = similarity_memory()
    assert isinstance(memory, Memory)


def test_similarity_memory_store_is_qdrant(
    mock_qdrant_client: MagicMock,
) -> None:
    memory = similarity_memory()
    assert isinstance(memory.store, QdrantMemoryStore)


def test_similarity_memory_max_results(mock_qdrant_client: MagicMock) -> None:
    memory = similarity_memory(max_results=7)
    assert memory.store.max_results == 7  # noqa: PLR2004


# --- reflective_memory ---


def test_reflective_memory_returns_memory(
    mock_qdrant_client: MagicMock,
) -> None:
    memory = reflective_memory(llm=make_llm())
    assert isinstance(memory, Memory)


def test_reflective_memory_store_is_qdrant(
    mock_qdrant_client: MagicMock,
) -> None:
    memory = reflective_memory(llm=make_llm())
    assert isinstance(memory.store, QdrantMemoryStore)


def test_reflective_memory_has_reflection_fn(
    mock_qdrant_client: MagicMock,
) -> None:
    memory = reflective_memory(llm=make_llm())
    assert "reflection" in memory.metadata_fns


def test_reflective_memory_max_results(mock_qdrant_client: MagicMock) -> None:
    memory = reflective_memory(llm=make_llm(), max_results=2)
    assert memory.store.max_results == 2  # noqa: PLR2004


@pytest.mark.asyncio
async def test_reflective_memory_reflect_calls_llm(
    mock_qdrant_client: MagicMock,
) -> None:
    llm = make_llm("always call the tool first")
    memory = reflective_memory(llm=llm)
    ep = make_episode()

    await memory.record(ep)

    llm.complete.assert_awaited_once()
    assert ep.metadata["reflection"] == "always call the tool first"


@pytest.mark.asyncio
async def test_reflective_memory_custom_template_used(
    mock_qdrant_client: MagicMock,
) -> None:
    custom = "Task: {instruction}\nResult: {result}\nCustom:"
    llm = make_llm("custom lesson")
    memory = reflective_memory(llm=llm, template=custom)
    ep = make_episode()

    await memory.record(ep)

    prompt_arg = llm.complete.call_args[0][0]
    assert "Custom:" in prompt_arg
