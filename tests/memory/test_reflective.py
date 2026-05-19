"""Unit tests for ReflectiveMemory."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.memory import BaseMemoryStore
from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.llm import CompleteResult
from llm_agents_from_scratch.data_structures.memory import Episode
from llm_agents_from_scratch.memory import ReflectiveMemory


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


def test_init_defaults() -> None:
    store = MagicMock(spec=BaseMemoryStore)
    llm = make_llm()
    memory = ReflectiveMemory(store=store, llm=llm)

    assert memory.store is store
    assert memory.llm is llm
    assert memory.n == 3  # noqa: PLR2004


def test_init_custom_n() -> None:
    store = MagicMock(spec=BaseMemoryStore)
    llm = make_llm()
    memory = ReflectiveMemory(store=store, llm=llm, n=5)

    assert memory.n == 5  # noqa: PLR2004


@pytest.mark.asyncio
async def test_record_calls_llm_complete() -> None:
    store = AsyncMock(spec=BaseMemoryStore)
    llm = make_llm()
    memory = ReflectiveMemory(store=store, llm=llm)
    ep = make_episode()

    await memory.record(ep)

    llm.complete.assert_awaited_once()
    prompt_arg = llm.complete.call_args[0][0]
    assert ep.task.instruction in prompt_arg
    assert ep.result.content in prompt_arg


@pytest.mark.asyncio
async def test_record_stores_episode_with_reflection() -> None:
    store = AsyncMock(spec=BaseMemoryStore)
    reflection = "Always verify the name first."
    llm = make_llm(reflection)
    memory = ReflectiveMemory(store=store, llm=llm)
    ep = make_episode()

    await memory.record(ep)

    store.write.assert_awaited_once()
    written: Episode = store.write.call_args[0][0]
    assert written.additional_data is not None
    assert written.additional_data["reflection"] == reflection


@pytest.mark.asyncio
async def test_summary() -> None:
    store = AsyncMock(spec=BaseMemoryStore)
    store.summary = AsyncMock(return_value="JSONMemoryStore: 2 episodes")
    llm = make_llm()
    memory = ReflectiveMemory(store=store, llm=llm, n=4)

    result = await memory.summary()

    assert "ReflectiveMemory" in result
    assert "4" in result
    assert "JSONMemoryStore" in result


def test_episode_str_renders_additional_data_as_xml_tags() -> None:
    task = Task(instruction="look up pikachu")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="pikachu is electric"),
        additional_data={"reflection": "Always verify spelling."},
    )

    output = str(ep)

    assert "<reflection>Always verify spelling.</reflection>" in output


def test_episode_str_no_extra_tags_when_additional_data_none() -> None:
    task = Task(instruction="look up pikachu")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="pikachu is electric"),
    )

    output = str(ep)

    assert "<reflection>" not in output
    assert "additional_data" not in output


def test_episode_format_concat_includes_additional_data() -> None:
    task = Task(instruction="look up pikachu")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="pikachu is electric"),
        additional_data={"reflection": "Always verify spelling."},
    )

    text = ep.format(mode="concat")

    assert "look up pikachu" in text
    assert "pikachu is electric" in text
    assert "Always verify spelling." in text


def test_episode_format_concat_excludes_completed_at_by_default() -> None:
    task = Task(instruction="look up pikachu")
    ep = Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="pikachu is electric"),
    )

    text = ep.format(mode="concat")

    assert "completed_at" not in text
    ts = ep.completed_at.strftime("%Y-%m-%d")
    assert ts not in text
