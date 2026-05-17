import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_agents_from_scratch.agent import LLMAgent
from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.memory import BaseMemory
from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import Episode
from llm_agents_from_scratch.data_structures.agent import (
    Task,
    TaskResult,
    TaskStep,
)
from llm_agents_from_scratch.errors import LLMAgentError, MaxStepsReachedError
from llm_agents_from_scratch.skills.constants import (
    EXPLICIT_SKILL_ACTIVATION_TEMPLATE,
    EXPLICIT_SKILL_ACTIVATION_WITH_PROMPT_TEMPLATE,
)


def test_init(mock_llm: BaseLLM) -> None:
    """Tests init of LLMAgent."""
    agent = LLMAgent(llm=mock_llm)

    assert agent.llm == mock_llm
    assert len(agent.tools) == 0


def test_init_raises_error_duplicated_tools(
    mock_llm: BaseLLM,
    _test_tool: BaseTool,
) -> None:
    """Tests init of LLMAgent."""
    with pytest.raises(LLMAgentError):
        LLMAgent(llm=mock_llm, tools=[_test_tool, _test_tool])


def test_add_tool(mock_llm: BaseLLM) -> None:
    """Tests add tool."""
    # arrange
    tool = MagicMock()
    agent = LLMAgent(llm=mock_llm)

    # act
    agent.add_tool(tool)

    # assert
    assert agent.tools == [tool]


def test_add_tool_raises_error(
    mock_llm: BaseLLM,
    _test_tool: BaseTool,
) -> None:
    """Tests add tool."""

    # arrange
    agent = LLMAgent(llm=mock_llm, tools=[_test_tool])

    with pytest.raises(LLMAgentError):
        agent.add_tool(_test_tool)


@pytest.mark.asyncio
@patch.object(LLMAgent.TaskHandler, "get_next_step")
async def test_run(
    mock_get_next_step: AsyncMock,
    mock_llm: BaseLLM,
) -> None:
    """Tests run method."""

    # arrange mocks
    task = Task(instruction="mock instruction")
    agent = LLMAgent(llm=mock_llm)

    mock_get_next_step.side_effect = [
        TaskStep(task_id=task.id_, instruction="mock step"),
        TaskStep(task_id=task.id_, instruction="another mock step"),
        TaskResult(task_id=task.id_, content="mock result"),
    ]

    # arrange
    agent = LLMAgent(llm=mock_llm)

    # act
    handler = agent.run(task)
    await handler

    # cleanup
    handler.background_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await handler.background_task
    assert str(handler.result()) == "mock result"
    expected_rollout = (
        "=== Task Step Start ===\n\n"
        "💬 assistant: My current instruction is 'mock step'\n\n"
        "💬 assistant: mock chat response\n\n"
        "=== Task Step End ===\n\n"
        "=== Task Step Start ===\n\n"
        "💬 assistant: My current instruction is 'another mock step'\n\n"
        "💬 assistant: mock chat response\n\n"
        "=== Task Step End ==="
    )
    assert handler.rollout == expected_rollout


@pytest.mark.asyncio
@patch.object(LLMAgent.TaskHandler, "get_next_step")
async def test_run_exception(
    mock_get_next_step: AsyncMock,
    mock_llm: BaseLLM,
) -> None:
    """Tests run method with exception."""
    err = RuntimeError("mock error")
    mock_get_next_step.side_effect = err

    # arrange
    agent = LLMAgent(llm=mock_llm)
    task = Task(instruction="mock instruction")

    # act
    handler = agent.run(task)
    await asyncio.sleep(0.1)  # Let it run

    assert handler.exception() == err


@pytest.mark.asyncio
@patch.object(LLMAgent.TaskHandler, "get_next_step")
async def test_run_max_steps_reached_error(
    mock_get_next_step: AsyncMock,
    mock_llm: BaseLLM,
) -> None:
    """Tests run method reaches max step."""

    # arrange
    task = Task(instruction="mock instruction")
    mock_get_next_step.side_effect = [
        TaskStep(
            instruction="mock 1",
            task_id=task.id_,
        ),
        TaskStep(
            instruction="mock 2",
            task_id=task.id_,
        ),
    ]
    agent = LLMAgent(llm=mock_llm)

    # act
    handler = agent.run(task, max_steps=1)
    await asyncio.sleep(0.1)  # Let it run

    assert isinstance(handler.exception(), MaxStepsReachedError)


def test_run_with_skill_no_prompt(mock_llm: BaseLLM) -> None:
    """Tests run_with_skill builds correct instruction without a prompt."""
    agent = LLMAgent(llm=mock_llm)

    with patch.object(agent, "run") as mock_run:
        agent.run_with_skill("summarize")

    task = mock_run.call_args.kwargs["task"]
    expected = EXPLICIT_SKILL_ACTIVATION_TEMPLATE.format(name="summarize")
    assert task.instruction == expected


def test_run_with_skill_with_prompt(mock_llm: BaseLLM) -> None:
    """Tests run_with_skill builds correct instruction with a prompt."""
    agent = LLMAgent(llm=mock_llm)

    with patch.object(agent, "run") as mock_run:
        agent.run_with_skill("summarize", prompt="Summarize this doc")

    task = mock_run.call_args.kwargs["task"]
    expected = EXPLICIT_SKILL_ACTIVATION_WITH_PROMPT_TEMPLATE.format(
        name="summarize",
        prompt="Summarize this doc",
    )
    assert task.instruction == expected


# Memory record tests (Chapter 7)


@pytest.mark.asyncio
@patch.object(LLMAgent.TaskHandler, "get_next_step")
async def test_run_records_episode_on_success(
    mock_get_next_step: AsyncMock,
    mock_llm: BaseLLM,
) -> None:
    """Tests memory.record is called with the completed Episode on success."""
    task = Task(instruction="mock instruction")
    task_result = TaskResult(task_id=task.id_, content="mock result")
    mock_get_next_step.side_effect = [task_result]

    mock_memory = AsyncMock(spec=BaseMemory)
    mock_memory.recall.return_value = ""
    agent = LLMAgent(llm=mock_llm, memories=[mock_memory])

    handler = agent.run(task)
    await handler

    mock_memory.record.assert_awaited_once()
    ep: Episode = mock_memory.record.call_args[0][0]
    assert isinstance(ep, Episode)
    assert ep.task == task
    assert ep.result == task_result


@pytest.mark.asyncio
@patch.object(LLMAgent.TaskHandler, "get_next_step")
async def test_run_records_episode_on_failure(
    mock_get_next_step: AsyncMock,
    mock_llm: BaseLLM,
) -> None:
    """Tests memory.record is called with a synthetic Episode on failure."""
    err = RuntimeError("boom")
    mock_get_next_step.side_effect = err

    mock_memory = AsyncMock(spec=BaseMemory)
    mock_memory.recall.return_value = ""
    task = Task(instruction="mock instruction")
    agent = LLMAgent(llm=mock_llm, memories=[mock_memory])

    agent.run(task)
    await asyncio.sleep(0.1)

    mock_memory.record.assert_awaited_once()
    ep: Episode = mock_memory.record.call_args[0][0]
    assert isinstance(ep, Episode)
    assert ep.task == task
    assert str(err) in ep.result.content


@pytest.mark.asyncio
@patch.object(LLMAgent.TaskHandler, "get_next_step")
async def test_run_records_episode_for_each_memory(
    mock_get_next_step: AsyncMock,
    mock_llm: BaseLLM,
) -> None:
    """Tests memory.record is called on every memory in agent.memories."""
    task = Task(instruction="mock instruction")
    task_result = TaskResult(task_id=task.id_, content="mock result")
    mock_get_next_step.side_effect = [task_result]

    mock_memory_a = AsyncMock(spec=BaseMemory)
    mock_memory_a.recall.return_value = ""
    mock_memory_b = AsyncMock(spec=BaseMemory)
    mock_memory_b.recall.return_value = ""
    agent = LLMAgent(llm=mock_llm, memories=[mock_memory_a, mock_memory_b])

    handler = agent.run(task)
    await handler

    mock_memory_a.record.assert_awaited_once()
    mock_memory_b.record.assert_awaited_once()
