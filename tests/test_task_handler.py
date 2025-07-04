import asyncio
from unittest.mock import AsyncMock

import pytest

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.core import TaskHandler
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ChatRole,
    Task,
    TaskStep,
    ToolCall,
)
from llm_agents_from_scratch.errors import TaskHandlerError


def test_task_handler_init(
    mock_llm: BaseLLM,
) -> None:
    handler = TaskHandler(
        task=Task(instruction="mock instruction"),
        llm=mock_llm,
        tools=[],
    )

    assert handler.task.instruction == "mock instruction"
    assert handler.llm == mock_llm
    assert handler.tools_registry == {}


def test_task_handler_raises_error_when_getting_unset_bg_task(
    mock_llm: BaseLLM,
) -> None:
    handler = TaskHandler(
        task=Task(instruction="mock instruction"),
        llm=mock_llm,
        tools=[],
    )

    with pytest.raises(TaskHandlerError):
        handler.background_task  # noqa: B018


@pytest.mark.asyncio
async def test_task_handler_raises_error_when_setting_already_set_bg_task(
    mock_llm: BaseLLM,
) -> None:
    async def fn() -> None:
        await asyncio.sleep(0.1)

    handler = TaskHandler(
        task=Task(instruction="mock instruction"),
        llm=mock_llm,
        tools=[],
    )

    handler.background_task = asyncio.create_task(fn())
    with pytest.raises(TaskHandlerError):
        handler.background_task = asyncio.create_task(fn())


@pytest.mark.asyncio
async def test_get_next_step(mock_llm: BaseLLM) -> None:
    """Tests get next step."""

    handler = TaskHandler(
        task=Task(instruction="mock instruction"),
        llm=mock_llm,
        tools=[],
    )

    # initial task step
    initial_step = await handler.get_next_step()

    # update rollout and get next step
    expected_next_step = TaskStep(
        instruction="Some next instruction.",
        last_step=False,
    )
    magic_mock_llm = AsyncMock()
    magic_mock_llm.structured_output.return_value = expected_next_step
    handler.llm = magic_mock_llm
    handler.rollout = "some progress"
    next_step = await handler.get_next_step()

    assert initial_step.instruction == "mock instruction"
    assert initial_step.last_step is False
    assert next_step == expected_next_step


@pytest.mark.asyncio
async def test_get_next_step_raises_error(mock_llm: BaseLLM) -> None:
    """Tests get next step."""

    handler = TaskHandler(
        task=Task(instruction="mock instruction"),
        llm=mock_llm,
        tools=[],
    )

    # initial task step
    initial_step = await handler.get_next_step()

    # update rollout and get next step
    magic_mock_llm = AsyncMock()
    magic_mock_llm.structured_output.side_effect = RuntimeError("oops.")
    handler.llm = magic_mock_llm
    handler.rollout = "some progress"

    with pytest.raises(
        TaskHandlerError,
        match="Failed to get next step: oops.",
    ):
        await handler.get_next_step()

    assert initial_step.instruction == "mock instruction"
    assert initial_step.last_step is False


def test_private_rollout_contribution_from_single_run_step(
    mock_llm: BaseLLM,
) -> None:
    """Tests helper method to get rollout contribution from run step."""
    handler = TaskHandler(
        task=Task(instruction="mock instruction"),
        llm=mock_llm,
        tools=[],
    )
    chat_history = [
        ChatMessage(
            role=ChatRole.SYSTEM,
            content="a system message",
        ),
        ChatMessage(
            role=ChatRole.USER,
            content="a user message",
        ),
        ChatMessage(
            role=ChatRole.ASSISTANT,
            content="an assistant message",
            tool_calls=[
                ToolCall(
                    tool_name="a tool",
                    arguments={"tool_arg": 1},
                ),
            ],
        ),
        ChatMessage(
            role=ChatRole.TOOL,
            content="\n\ttool name: `a tool`\n\ttool result: 1+2=3.",
        ),
        ChatMessage(
            role=ChatRole.ASSISTANT,
            content="done!",
        ),
    ]

    # act
    rollout_contribution = handler._rollout_contribution_from_single_run_step(
        chat_history=chat_history,
    )

    expected_rollout_contribution = (
        "user: a user message\n"
        "assistant: an assistant message\n"
        "tool: \n\ttool name: `a tool`\n\ttool result: 1+2=3.\n"
        "assistant: done!"
    )
    assert rollout_contribution == expected_rollout_contribution
