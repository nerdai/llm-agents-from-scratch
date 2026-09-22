"""Regression coverage for LLMAgentA2AExecutor.cancel()'s ordering.

The existing unit tests in test_executor.py call `executor.cancel()`
directly against a bare `EventQueueLegacy` -- nothing in that path
ever closes the queue, so those tests pass under any ordering of
`cancel()`'s internals, including a broken one. They cannot catch a
regression in the invariant `cancel()` actually depends on: publish
the terminal status before anything in the method can suspend.

These tests instead drive the real SDK request-handler stack
end-to-end (``DefaultRequestHandler`` -> its producer loop -> a real
event queue -> ``InMemoryTaskStore``), which is what actually closes
the queue once the SDK cancels the producer running ``execute()``.
Only the LLM is a fake, and only so a run is slow enough to cancel
mid-step.

They also check the other half of what cancel() promises, which
test_executor.py's existing test only checks against the same bare
queue: that the in-flight work actually stopped (background_task
settles), not just that a status update claiming so got published.

a2a-sdk 1.1.4 added a backstop for the task *store* specifically:
``ActiveTask.cancel()`` (``a2a/server/agent_execution/active_task.py``)
now waits for the producer to fully settle and, if the task is not
already in a terminal state, force-writes ``TASK_STATE_CANCELED``
straight to the task store -- bypassing the event queue entirely, so
a dropped publish from our own ``cancel()`` no longer prevents the
store from reaching CANCELED. Per that method's own docstring, this
backstop covers the store only: "That write is not guaranteed to
reach an active subscriber stream." The publish-before-anything-that-
can-suspend ordering in our ``cancel()`` therefore stays correct
practice -- it is what gets a live streaming subscriber the CANCELED
status promptly, which the backstop does not provide -- but it is no
longer load-bearing for what this test suite can observe through
``InMemoryTaskStore``. See
``test_cancel_drops_status_if_anything_yields_before_publish`` for
where that shows up.
"""

import asyncio
import contextlib
import dataclasses
import logging
from importlib import metadata
from typing import Any, Sequence

import pytest
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types.a2a_pb2 import (
    CancelTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)
from a2a.utils.errors import TaskNotFoundError

from llm_agents_from_scratch.a2a.server.executor import (
    LLMAgentA2AExecutor,
    build_agent_card,
)
from llm_agents_from_scratch.agent.llm_agent import LLMAgent
from llm_agents_from_scratch.base.llm import BaseLLM, StructuredOutputType
from llm_agents_from_scratch.base.tool import Tool
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    CompleteResult,
    ToolCallResult,
)

_A2A_SDK_VERSION = tuple(
    int(part) for part in metadata.version("a2a-sdk").split(".")[:3]
)
_HAS_CANCEL_STORE_BACKSTOP = _A2A_SDK_VERSION >= (1, 1, 4)


class _ParkingLLM(BaseLLM):
    """An LLM whose chat() never returns, so a run is cancellable mid-step."""

    async def complete(self, prompt: str, **kwargs: Any) -> CompleteResult:
        return CompleteResult(response="mock", full_response="mock")

    async def structured_output(
        self,
        prompt: str,
        mdl: type[StructuredOutputType],
        **kwargs: Any,
    ) -> StructuredOutputType:
        return mdl()

    async def chat(
        self,
        input: str,
        chat_history: Sequence[ChatMessage] | None = None,
        tools: Sequence[Tool] | None = None,
        **kwargs: Any,
    ) -> tuple[ChatMessage, ChatMessage]:
        await asyncio.sleep(300)
        raise AssertionError("unreachable")

    async def continue_chat_with_tool_results(
        self,
        tool_call_results: Sequence[ToolCallResult],
        chat_history: Sequence[ChatMessage],
        tools: Sequence[Tool] | None = None,
        **kwargs: Any,
    ) -> tuple[list[ChatMessage], ChatMessage]:
        raise AssertionError("unreachable")


class _YieldsBeforePublishExecutor(LLMAgentA2AExecutor):
    """cancel() with a single bare yield inserted before the publish.

    A full override, not a wrap-and-delegate: the yield must land
    *after* the task_handler lookup/validation and *before*
    ``updater.cancel()``, matching the shipped method's structure
    exactly except for that one inserted yield. Wrapping
    ``super().cancel()`` with a yield in front would yield before the
    lookup too, so the producer's own cleanup pops the task_handlers
    entry first and this raises TaskNotFoundError instead of
    exercising the silent-drop path this test means to prove.

    Negative control: proves the harness below actually detects a
    broken ordering, rather than trivially passing regardless of
    what cancel() does. On a2a-sdk >= 1.1.4 this no longer produces
    an observable failure through the task store -- see this test
    file's module docstring for why.
    """

    async def cancel(self, context: Any, event_queue: Any) -> None:
        if context.task_id is None or context.context_id is None:
            raise ValueError(
                "RequestContext is missing a task_id or context_id.",
            )
        task_handler = self._task_handlers.get(context.task_id)
        if task_handler is None:
            raise TaskNotFoundError(
                f"No in-flight task found for id '{context.task_id}'.",
            )

        await asyncio.sleep(0)  # <-- the only difference from cancel()

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()

        task_handler.background_task.cancel()
        try:  # noqa: SIM105
            await task_handler.background_task
        except asyncio.CancelledError:
            pass
        if not task_handler.done():
            task_handler.cancel()


@dataclasses.dataclass
class _CancelOutcome:
    """Result of one send-then-cancel run.

    Attributes:
        persisted_state: The terminal ``TaskState`` name persisted to
            the task store.
        dropped: How many times the SDK logged its silent-drop
            warning for an enqueue onto an already-closed queue.
        handler_done: Whether the tracked ``TaskHandler`` Future
            settled.
        handler_cancelled: Whether it settled specifically via
            cancellation.
        background_task_done: Whether the ``asyncio.Task`` actually
            running the agent loop finished -- the real signal that
            the in-flight work stopped, as opposed to the Future
            wrapper merely reporting cancelled.
    """

    persisted_state: str
    dropped: int
    handler_done: bool
    handler_cancelled: bool
    background_task_done: bool


async def _run_and_cancel(
    executor: LLMAgentA2AExecutor,
) -> _CancelOutcome:
    """Sends a task through the real SDK stack, then cancels it."""

    class _DropWatcher(logging.Handler):
        """Counts enqueue-side "queue closed" drops, not exact wording.

        a2a-sdk's EventQueue.enqueue_event() logs one of two slightly
        different messages depending on *when* it discovers the queue
        is closed ("Event will not be enqueued." vs ", during
        enqueuing. Event dropped."), and a dequeue_event() warning
        ("will not be dequeued") uses near-identical phrasing for an
        unrelated event. Matching the two substrings both enqueue
        warnings share ("closed" and "enqueu", present in "enqueued"/
        "enqueuing" but not "dequeued") catches either wording without
        pinning to the exact copy, which could change across SDK
        versions, while still excluding the dequeue warning.
        """

        def __init__(self) -> None:
            super().__init__(level=logging.DEBUG)
            self.dropped = 0

        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if "closed" in message and "enqueu" in message:
                self.dropped += 1

    watcher = _DropWatcher()
    logger = logging.getLogger("a2a")
    previous_level = logger.level
    previous_disable = logging.root.manager.disable
    logger.addHandler(watcher)
    logger.setLevel(logging.DEBUG)
    # tests/conftest.py's autouse suppress_logging fixture disables
    # logging below CRITICAL for every test -- lift that locally so
    # the SDK's drop warning actually reaches our handler.
    logging.disable(logging.NOTSET)

    store = InMemoryTaskStore()
    card = build_agent_card(
        name="cancel-regression",
        description="cancel-regression",
        url="http://localhost:9999/",
    )
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=store,
        agent_card=card,
    )
    ctx = ServerCallContext()

    send_request = SendMessageRequest(
        message=Message(
            message_id="m1",
            role=Role.ROLE_USER,
            parts=[Part(text="hang forever")],
        ),
    )
    send_task = asyncio.create_task(handler.on_message_send(send_request, ctx))

    for _ in range(2000):
        await asyncio.sleep(0.001)
        if executor._task_handlers:
            break
    else:
        raise AssertionError("Task never registered with the executor.")
    await asyncio.sleep(0.02)

    task_id = next(iter(executor._task_handlers))
    task_handler = executor._task_handlers[task_id]

    try:
        await asyncio.wait_for(
            handler.on_cancel_task(CancelTaskRequest(id=task_id), ctx),
            timeout=10,
        )
    finally:
        send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, TimeoutError):
            await asyncio.wait_for(send_task, timeout=5)
        await handler.aclose()
        logger.removeHandler(watcher)
        logger.setLevel(previous_level)
        logging.disable(previous_disable)

    persisted = await store.get(task_id, ctx)
    if persisted is None:
        # A harness failure (e.g. the task store never saw the initial
        # submission) is not the silent-drop behavior under test --
        # fail loudly here instead of letting a "MISSING" sentinel
        # satisfy `!= "TASK_STATE_CANCELED"` in the negative-control
        # test for the wrong reason.
        raise AssertionError(
            f"Task {task_id!r} was never persisted to the task store.",
        )
    persisted_state = TaskState.Name(persisted.status.state)
    return _CancelOutcome(
        persisted_state=persisted_state,
        dropped=watcher.dropped,
        handler_done=task_handler.done(),
        handler_cancelled=task_handler.cancelled(),
        background_task_done=task_handler.background_task.done(),
    )


@pytest.mark.asyncio
async def test_cancel_persists_canceled_via_real_sdk_producer_loop() -> None:
    """Tests the shipped cancel() ordering survives the real SDK stack.

    Unlike test_executor.py's cancel() tests, this drives the actual
    DefaultRequestHandler producer loop, which is what closes the
    event_queue once the SDK cancels execute()'s producer task --
    the mechanism cancel()'s ordering has to survive.

    Checks both halves of what cancel() promises: the terminal status
    lands (persisted_state, dropped), and the in-flight work actually
    stopped (background_task_done) rather than being left running
    orphaned underneath a status update that merely claims it's gone.
    """
    agent = LLMAgent(llm=_ParkingLLM())
    executor = LLMAgentA2AExecutor(agent=agent)

    outcome = await _run_and_cancel(executor)

    assert outcome.persisted_state == "TASK_STATE_CANCELED"
    assert outcome.dropped == 0
    assert outcome.handler_done
    assert outcome.handler_cancelled
    assert outcome.background_task_done


@pytest.mark.asyncio
@pytest.mark.xfail(
    _HAS_CANCEL_STORE_BACKSTOP,
    strict=True,
    reason=(
        "a2a-sdk >= 1.1.4's ActiveTask.cancel() force-writes "
        "TASK_STATE_CANCELED to the task store once the producer "
        "settles, even if our own cancel() dropped its publish -- "
        "see this file's module docstring. strict=True so this turns "
        "into a hard failure (XPASS) if a future a2a-sdk version "
        "removes that backstop and the old drop becomes observable "
        "again, which is the signal to revisit this xfail."
    ),
)
async def test_cancel_drops_status_if_anything_yields_before_publish() -> None:
    """Negative control: a yield before the publish loses the status.

    Confirms the harness above would actually fail if cancel()'s
    ordering regressed -- without this, a passing positive test alone
    wouldn't prove the harness detects the failure mode it exists for.
    True on a2a-sdk < 1.1.4; neutralized by the task-store backstop on
    1.1.4+, hence the xfail above rather than deleting the test.

    handler_done/handler_cancelled/background_task_done still hold
    here: the SDK's ActiveTask.cancel() cancels the producer task (and
    so task_handler, and so background_task) *before* calling our
    cancel() at all -- confirmed in the ordering investigation this
    test file follows up on. A broken publish ordering only breaks
    status *reporting*; it doesn't stop the underlying work from
    actually being torn down, since that happens independently of
    anything our cancel() body does on this SDK path.
    """
    agent = LLMAgent(llm=_ParkingLLM())
    executor = _YieldsBeforePublishExecutor(agent=agent)

    outcome = await _run_and_cancel(executor)

    assert outcome.persisted_state != "TASK_STATE_CANCELED"
    assert outcome.dropped >= 1
    assert outcome.handler_done
    assert outcome.handler_cancelled
    assert outcome.background_task_done
