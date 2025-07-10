import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_agents_from_scratch.agent import LLMAgent, TaskHandler
from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures.agent import (
    Task,
    TaskResult,
    TaskStep,
)


def test_init(mock_llm: BaseLLM) -> None:
    """Tests init of LLMAgent."""
    agent = LLMAgent(llm=mock_llm)

    assert len(agent.tools) == 0
    assert agent.llm == mock_llm


def test_add_tool(mock_llm: BaseLLM) -> None:
    """Tests add tool."""
    # arrange
    tool = MagicMock()
    agent = LLMAgent(llm=mock_llm)

    # act
    agent.add_tool(tool)

    # assert
    assert agent.tools == [tool]


@pytest.mark.asyncio
@patch.object(TaskHandler, "get_next_step")
async def test_run(
    mock_get_next_step: AsyncMock,
    mock_llm: BaseLLM,
) -> None:
    """Tests run method."""

    # arrange mocks
    task = Task(instruction="mock instruction")
    agent = LLMAgent(llm=mock_llm)

    mock_get_next_step.side_effect = [
        TaskStep(instruction="mock step"),
        TaskResult(task=task, content="mock result"),
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
    expected_rollout = "assistant: mock step\nassistant: mock chat response"
    assert handler.result().rollout == expected_rollout


@pytest.mark.asyncio
@patch.object(TaskHandler, "get_next_step")
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
