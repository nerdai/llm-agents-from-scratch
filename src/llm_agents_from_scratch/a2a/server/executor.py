"""LLMAgentA2AExecutor — bridges inbound A2A tasks to an LLMAgent."""

import asyncio

from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from a2a.utils.errors import TaskNotFoundError

from llm_agents_from_scratch.agent.llm_agent import LLMAgent
from llm_agents_from_scratch.data_structures import Task


class LLMAgentA2AExecutor(AgentExecutor):
    """Bridges inbound A2A tasks to an ``LLMAgent``.

    Per request: ``RequestContext`` -> ``Task(instruction=...)`` ->
    ``await agent.run()`` -> ``Artifact``. Results live in
    ``task.artifacts``, not in messages — the A2A v1.0 spec reserves
    messages for communication and artifacts for outputs.

    Every dispatch runs to completion or failure in one shot; there is
    no ``input_required`` support on this (server) side. §8.5's channel
    problem was already answered in Ch9 by dissolution (sub-agents run
    in-process, same terminal), so implementing a non-terminal
    ``SendMessage`` return here would mostly pay off a forward
    reference already paid — a named skip, alongside streaming,
    ``contextId``, auth, and push notifications. This does not affect
    ``UseA2AAgentTool`` (client side): resuming a task a *peer* parked
    in ``input_required`` is a different, much cheaper problem, and is
    in scope there.

    No concurrency limit: the SDK invokes ``execute()`` once per
    task_id with no cap on how many run at once — two clients dispatching
    at the same moment get two independent ``ActiveTask``s and two
    genuinely concurrent calls into this executor (the SDK only
    serializes repeat calls for the *same* task_id, not across
    different ones). Nothing here bounds how many concurrent
    ``agent.run()`` calls fan out to the backing LLM. Production use
    serving real traffic should guard this — e.g. an ``asyncio.
    Semaphore`` acquired at the top of ``execute()`` — left as a
    callout rather than built in, to keep this an educational
    framework rather than a production-hardened one.

    Attributes:
        agent (LLMAgent): The agent this executor serves.
    """

    def __init__(self, agent: LLMAgent) -> None:
        """Initialise with the agent to serve.

        Args:
            agent (LLMAgent): The agent to bridge inbound A2A tasks to.
        """
        self.agent = agent
        self._task_handlers: dict[str, LLMAgent.TaskHandler] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Runs the agent on the caller's instruction to completion.

        Only ``Exception`` is caught below, deliberately not
        ``BaseException``: ``asyncio.CancelledError`` (raised here when
        ``cancel()`` cancels ``task_handler``) must propagate uncaught.
        The SDK's own producer loop wrapping this call has its own
        ``except CancelledError`` that closes the event queue and lets
        the producer task actually terminate. Catching and returning
        normally here instead would make ``execute()`` look like it
        finished successfully, leaving that producer task running —
        parked awaiting a request that will never come.

        Args:
            context (RequestContext): The request context containing
                the task instruction.
            event_queue (EventQueue): The queue to publish task
                status/artifact events to.
        """
        if context.current_task:
            task = context.current_task
        else:
            if context.message is None:
                raise ValueError(
                    "RequestContext is missing the user's Message.",
                )
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.submit()
        await updater.start_work()

        instruction = context.get_user_input()
        task_handler = self.agent.run(Task(instruction=instruction))
        self._task_handlers[task.id] = task_handler
        try:
            result = await task_handler
        except Exception as e:
            await updater.update_status(
                TaskState.TASK_STATE_FAILED,
                message=updater.new_agent_message([new_text_part(str(e))]),
            )
            return
        finally:
            self._task_handlers.pop(task.id, None)

        await updater.add_artifact(
            parts=[new_text_part(result.content)],
            name="task_result",
        )
        await updater.complete()

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancels the agent run backing an in-flight task.

        Args:
            context (RequestContext): The request context naming the
                task to cancel.
            event_queue (EventQueue): The queue to publish the
                cancellation status update to.

        Raises:
            TaskNotFoundError: If ``context.task_id`` names no
                in-flight task on this executor.
        """
        if context.task_id is None or context.context_id is None:
            raise ValueError(
                "RequestContext is missing a task_id or context_id.",
            )
        task_handler = self._task_handlers.get(context.task_id)
        if task_handler is None:
            raise TaskNotFoundError(
                f"No in-flight task found for id '{context.task_id}'.",
            )

        # Publish CANCELED before actually cancelling task_handler.
        # Cancelling task_handler first would wake execute()'s
        # `await task_handler` immediately, propagating
        # CancelledError and triggering the SDK's own producer-loop
        # cleanup, which closes this same event_queue -- racing our
        # own enqueue below and sometimes winning, silently dropping
        # the terminal status update (confirmed live: the queue
        # closing mid-enqueue leaves the task's persisted state stuck
        # at its last non-terminal value instead of CANCELED).
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()

        # TaskHandler is a plain asyncio.Future manually driven by
        # _process_loop -- cancelling background_task alone stops the
        # work but never settles task_handler itself (its except
        # Exception clause doesn't catch CancelledError), so anything
        # awaiting it would hang. Settle both, in order.
        task_handler.background_task.cancel()
        try:  # noqa: SIM105
            await task_handler.background_task
        except asyncio.CancelledError:
            pass
        if not task_handler.done():
            task_handler.cancel()
