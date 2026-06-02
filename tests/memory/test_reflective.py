"""Unit tests for ReflectiveMemory recipe."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.llm import CompleteResult
from llm_agents_from_scratch.data_structures.memory import Episode, RecallMode
from llm_agents_from_scratch.memory import Memory, ReflectiveMemory
from llm_agents_from_scratch.memory_stores.json import JSONMemoryStore


def make_episode(
    instruction: str = "look up pikachu",
    content: str = "pikachu is electric",
) -> Episode:
    task = Task(instruction=instruction)
    return Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content=content),
    )


def make_llm(reflection: str = "Always verify the name first.") -> BaseLLM:
    llm = MagicMock(spec=BaseLLM)
    llm.complete = AsyncMock(
        return_value=CompleteResult(response=reflection, prompt=""),
    )
    return llm


def test_returns_memory_instance(tmp_path: Path) -> None:
    memory = ReflectiveMemory(path=tmp_path, llm=make_llm())
    assert isinstance(memory, Memory)


def test_store_is_json(tmp_path: Path) -> None:
    memory = ReflectiveMemory(path=tmp_path, llm=make_llm())
    assert isinstance(memory.store, JSONMemoryStore)


def test_store_recall_mode_is_recent(tmp_path: Path) -> None:
    memory = ReflectiveMemory(path=tmp_path, llm=make_llm())
    assert memory.store.recall_mode == RecallMode.RECENT


def test_max_results_set_from_n(tmp_path: Path) -> None:
    memory = ReflectiveMemory(path=tmp_path, llm=make_llm(), n=7)
    assert memory.store.max_results == 7  # noqa: PLR2004


def test_has_reflection_metadata_fn(tmp_path: Path) -> None:
    memory = ReflectiveMemory(path=tmp_path, llm=make_llm())
    assert "reflection" in memory.metadata_fns


@pytest.mark.asyncio
async def test_record_calls_llm_and_stores_reflection(tmp_path: Path) -> None:
    reflection = "Always verify the name first."
    llm = make_llm(reflection)
    memory = ReflectiveMemory(path=tmp_path, llm=llm)
    ep = make_episode()

    await memory.record(ep)

    llm.complete.assert_awaited_once()
    prompt_arg = llm.complete.call_args[0][0]
    assert ep.task.instruction in prompt_arg
    assert ep.result.content in prompt_arg
    assert ep.metadata["reflection"] == reflection


def test_episode_str_renders_metadata_as_xml_tags() -> None:
    task = Task(instruction="look up pikachu")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="pikachu is electric"),
        metadata={"reflection": "Always verify spelling."},
    )

    output = str(ep)

    assert "<reflection>Always verify spelling.</reflection>" in output


def test_episode_str_no_extra_tags_when_metadata_empty() -> None:
    task = Task(instruction="look up pikachu")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="pikachu is electric"),
    )

    output = str(ep)

    assert "<reflection>" not in output


def test_episode_format_concat_includes_metadata() -> None:
    task = Task(instruction="look up pikachu")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="pikachu is electric"),
        metadata={"reflection": "Always verify spelling."},
    )

    text = ep.format(mode="concat")

    assert "look up pikachu" in text
    assert "pikachu is electric" in text
    assert "Always verify spelling." in text


def test_episode_format_concat_default_includes_completed_at() -> None:
    task = Task(instruction="look up pikachu")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="pikachu is electric"),
    )

    text = ep.format(mode="concat")

    ts = ep.completed_at.strftime("%Y-%m-%d %H:%M:%S")
    assert ts in text
