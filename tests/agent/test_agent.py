import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_agents_from_scratch.agent import LLMAgent
from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures.agent import (
    Task,
    TaskResult,
    TaskStep,
)
from llm_agents_from_scratch.errors import LLMAgentError, MaxStepsReachedError


def test_init(mock_llm: BaseLLM) -> None:
    """Tests init of LLMAgent."""
    agent = LLMAgent(llm=mock_llm)

    assert len(agent.tools) == 0
    assert agent.llm == mock_llm


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
    assert str(handler.result().task_result) == "mock result"
    expected_rollout = (
        "assistant: The current instruction is 'mock step'\n"
        "assistant: mock chat response"
    )
    assert handler.result().rollout == expected_rollout


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
