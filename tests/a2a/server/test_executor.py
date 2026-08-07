"""Unit tests for LLMAgentA2AExecutor."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from a2a.server.events import EventQueueLegacy
from a2a.types import Message, Part, Role, TaskState
from a2a.utils.errors import TaskNotFoundError

from llm_agents_from_scratch import LLMAgent
from llm_agents_from_scratch.a2a.server.executor import LLMAgentA2AExecutor
from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures import Task, TaskResult, TaskStep


def _context(
    task_id: str = "t1",
    context_id: str = "c1",
    instruction: str = "do the thing",
    current_task: object | None = None,
) -> MagicMock:
    context = MagicMock()
    context.task_id = task_id
    context.context_id = context_id
    context.current_task = current_task
    context.message = Message(
        role=Role.ROLE_USER,
        parts=[Part(text=instruction)],
        task_id=task_id,
        context_id=context_id,
    )
    context.get_user_input.return_value = instruction
    return context


async def _drain(queue: EventQueueLegacy, timeout: float = 0.5) -> list:
    events = []
    try:
        while True:
            events.append(
                await asyncio.wait_for(queue.dequeue_event(), timeout=timeout),
            )
    except asyncio.TimeoutError:
        pass
    return events


@pytest.mark.asyncio
async def test_execute_completes_task(mock_llm: BaseLLM) -> None:
    """Tests a successful run publishes COMPLETED with the result artifact."""
    agent = LLMAgent(llm=mock_llm)
    executor = LLMAgentA2AExecutor(agent=agent)
    queue = EventQueueLegacy()
    context = _context()

    with patch.object(LLMAgent.TaskHandler, "get_next_step") as mock_next_step:
        mock_next_step.side_effect = [
            TaskStep(task_id="x", instruction="do the thing"),
            TaskResult(task_id="x", content="the answer"),
        ]
        await executor.execute(context, queue)

    events = await _drain(queue)

    statuses = [e.status.state for e in events if hasattr(e, "status")]
    assert TaskState.TASK_STATE_COMPLETED in statuses
    artifacts = [e for e in events if hasattr(e, "artifact")]
    assert len(artifacts) == 1
    assert artifacts[0].artifact.parts[0].text == "the answer"
    assert context.task_id not in executor._task_handlers


@pytest.mark.asyncio
async def test_execute_enqueues_new_task_when_no_current_task(
    mock_llm: BaseLLM,
) -> None:
    """Tests a Task event is enqueued when context.current_task is None."""
    agent = LLMAgent(llm=mock_llm)
    executor = LLMAgentA2AExecutor(agent=agent)
    queue = EventQueueLegacy()
    context = _context(current_task=None)

    with patch.object(LLMAgent.TaskHandler, "get_next_step") as mock_next_step:
        mock_next_step.side_effect = [
            TaskStep(task_id="x", instruction="do the thing"),
            TaskResult(task_id="x", content="the answer"),
        ]
        await executor.execute(context, queue)

    events = await _drain(queue)

    task_events = [e for e in events if type(e).__name__ == "Task"]
    assert len(task_events) == 1
    assert task_events[0].id == "t1"


@pytest.mark.asyncio
async def test_execute_skips_new_task_when_current_task_set(
    mock_llm: BaseLLM,
) -> None:
    """Tests no Task event is enqueued when context.current_task is set."""
    agent = LLMAgent(llm=mock_llm)
    executor = LLMAgentA2AExecutor(agent=agent)
    queue = EventQueueLegacy()
    context = _context(current_task=MagicMock())

    with patch.object(LLMAgent.TaskHandler, "get_next_step") as mock_next_step:
        mock_next_step.side_effect = [
            TaskStep(task_id="x", instruction="do the thing"),
            TaskResult(task_id="x", content="the answer"),
        ]
        await executor.execute(context, queue)

    events = await _drain(queue)

    task_events = [e for e in events if type(e).__name__ == "Task"]
    assert len(task_events) == 0


@pytest.mark.asyncio
async def test_execute_raises_on_missing_task_id(mock_llm: BaseLLM) -> None:
    """Tests a missing task_id raises before any dispatch."""
    executor = LLMAgentA2AExecutor(agent=LLMAgent(llm=mock_llm))
    context = _context(task_id=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="task_id or context_id"):
        await executor.execute(context, EventQueueLegacy())


@pytest.mark.asyncio
async def test_execute_raises_on_missing_message_when_no_current_task(
    mock_llm: BaseLLM,
) -> None:
    """Tests a missing Message raises when there's no current_task to resume."""
    executor = LLMAgentA2AExecutor(agent=LLMAgent(llm=mock_llm))
    context = _context()
    context.message = None

    with pytest.raises(ValueError, match="user's Message"):
        await executor.execute(context, EventQueueLegacy())


@pytest.mark.asyncio
async def test_execute_maps_exception_to_failed(mock_llm: BaseLLM) -> None:
    """Tests an exception during the run publishes FAILED, doesn't raise."""
    agent = LLMAgent(llm=mock_llm)
    executor = LLMAgentA2AExecutor(agent=agent)
    queue = EventQueueLegacy()
    context = _context()

    with patch.object(LLMAgent.TaskHandler, "get_next_step") as mock_next_step:
        mock_next_step.side_effect = RuntimeError("boom")
        await executor.execute(context, queue)

    events = await _drain(queue)

    statuses = [e.status.state for e in events if hasattr(e, "status")]
    assert TaskState.TASK_STATE_FAILED in statuses
    assert context.task_id not in executor._task_handlers


@pytest.mark.asyncio
async def test_cancel_raises_task_not_found_for_unknown_id(
    mock_llm: BaseLLM,
) -> None:
    """Tests cancelling an untracked task_id raises TaskNotFoundError."""
    executor = LLMAgentA2AExecutor(agent=LLMAgent(llm=mock_llm))
    context = _context(task_id="unknown")

    with pytest.raises(TaskNotFoundError):
        await executor.cancel(context, EventQueueLegacy())


@pytest.mark.asyncio
async def test_cancel_raises_on_missing_task_id(mock_llm: BaseLLM) -> None:
    """Tests a missing task_id raises before any lookup."""
    executor = LLMAgentA2AExecutor(agent=LLMAgent(llm=mock_llm))
    context = _context(task_id=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="task_id or context_id"):
        await executor.cancel(context, EventQueueLegacy())


@pytest.mark.asyncio
async def test_execute_returns_quietly_when_cancelled_mid_run(
    mock_llm: BaseLLM,
) -> None:
    """Tests execute() returns cleanly when cancel() interrupts its await."""
    agent = LLMAgent(llm=mock_llm)
    executor = LLMAgentA2AExecutor(agent=agent)
    queue = EventQueueLegacy()
    context = _context()

    async def _hang(*args: object, **kwargs: object) -> None:
        await asyncio.sleep(300)

    with patch.object(LLMAgent.TaskHandler, "get_next_step") as mock_next_step:
        mock_next_step.side_effect = _hang
        execute_task = asyncio.create_task(executor.execute(context, queue))
        await asyncio.sleep(0.1)

        await executor.cancel(context, EventQueueLegacy())
        await asyncio.wait_for(execute_task, timeout=1)

    assert execute_task.exception() is None


@pytest.mark.asyncio
async def test_cancel_cancels_in_flight_task(mock_llm: BaseLLM) -> None:
    """Tests cancel() settles the tracked TaskHandler and publishes CANCELED."""
    agent = LLMAgent(llm=mock_llm)
    executor = LLMAgentA2AExecutor(agent=agent)
    queue = EventQueueLegacy()

    task_handler = agent.run(Task(instruction="hang forever"))
    executor._task_handlers["t1"] = task_handler

    context = _context()
    await executor.cancel(context, queue)

    assert task_handler.done()
    assert task_handler.cancelled()

    events = await _drain(queue)
    statuses = [e.status.state for e in events if hasattr(e, "status")]
    assert TaskState.TASK_STATE_CANCELED in statuses
