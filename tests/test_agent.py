import asyncio
from unittest.mock import MagicMock, patch

import pytest
from typing_extensions import override

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.core import LLMAgent, TaskHandler
from llm_agents_from_scratch.data_structures.agent import (
    Task,
    TaskStep,
    TaskStepResult,
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
@patch("llm_agents_from_scratch.core.agent.TaskHandler")
async def test_run(
    mock_task_handler_class: MagicMock,
    mock_llm: BaseLLM,
) -> None:
    """Tests run method."""

    class MockTaskHandler(TaskHandler):
        @override
        async def get_next_step(self) -> TaskStep | None:
            await asyncio.sleep(0.1)
            return TaskStep(instruction="mock step")

        @override
        async def run_step(self, step: TaskStep) -> TaskStepResult:
            await asyncio.sleep(0.1)
            return TaskStepResult(
                task_step=step,
                content="mock result",
                last_step=True,
            )

    # arrange
    agent = LLMAgent(llm=mock_llm)
    task = Task(instruction="mock instruction")
    mock_handler = MockTaskHandler(task, agent.llm, agent.tools)
    mock_task_handler_class.return_value = mock_handler

    # act
    handler = agent.run(task)
    await handler

    # cleanup
    for t in handler._asyncio_tasks:
        t.cancel()

    assert handler == mock_handler
    mock_task_handler_class.assert_called_once_with(
        task,
        agent.llm,
        agent.tools,
    )
    assert handler.result().content == "mock result"


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.core.agent.TaskHandler")
async def test_run_exception(
    mock_task_handler_class: MagicMock,
    mock_llm: BaseLLM,
) -> None:
    """Tests run method with exception."""
    err = RuntimeError("mock error")

    class MockTaskHandler(TaskHandler):
        @override
        async def get_next_step(self) -> TaskStep | None:
            raise err

    # arrange
    agent = LLMAgent(llm=mock_llm)
    task = Task(instruction="mock instruction")
    mock_handler = MockTaskHandler(task, agent.llm, agent.tools)
    mock_task_handler_class.return_value = mock_handler

    # act
    handler = agent.run(task)
    await asyncio.sleep(0.1)  # Let it run

    assert handler == mock_handler
    mock_task_handler_class.assert_called_once_with(
        task,
        agent.llm,
        agent.tools,
    )
    assert handler.exception() == err
