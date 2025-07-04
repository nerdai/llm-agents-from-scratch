import asyncio

import pytest

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.core import TaskHandler
from llm_agents_from_scratch.data_structures.agent import Task
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
