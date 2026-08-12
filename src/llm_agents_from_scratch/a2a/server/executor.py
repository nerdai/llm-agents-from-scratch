"""LLMAgentA2AExecutor — bridges inbound A2A tasks to an LLMAgent."""

import asyncio

from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentCardSignature,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    SecurityRequirement,
    SecurityScheme,
    TaskState,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
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
    problem was already answered in Ch9 by dissolution (subagents run
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
        _task_handlers (dict[str, LLMAgent.TaskHandler]): In-flight
            runs, keyed by task_id. ``execute()`` and ``cancel()`` are
            separate calls with no shared local state connecting
            them — this registry is what lets ``cancel()`` find the
            specific ``TaskHandler`` a concurrent ``execute()`` call
            (for a different task_id) is driving. Entries are added
            when a run starts and removed in ``execute()``'s
            ``finally`` once that task settles.
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


def build_agent_card(  # noqa: PLR0913, PLR0917
    name: str,
    description: str,
    url: str,
    version: str = "0.1.0",
    skills: list[AgentSkill] | None = None,
    provider: AgentProvider | None = None,
    documentation_url: str | None = None,
    security_schemes: dict[str, SecurityScheme] | None = None,
    security_requirements: list[SecurityRequirement] | None = None,
    signatures: list[AgentCardSignature] | None = None,
    icon_url: str | None = None,
) -> AgentCard:
    """Builds an ``AgentCard`` for serving an ``LLMAgentA2AExecutor``.

    A plain function returning the SDK's own type rather than a class
    of ours, so readers keep the protocol's vocabulary. Necessarily
    opinionated, not a neutral general-purpose ``AgentCard``
    constructor: its job is a card that's honest about what
    ``LLMAgentA2AExecutor`` specifically does, so every field
    describing executor *behavior* is fixed rather than exposed as a
    parameter — ``supported_interfaces`` is always a single
    JSON-RPC/v1.0 entry at ``url`` (the only transport
    ``DefaultRequestHandler``/``LLMAgentA2AExecutor`` speak),
    ``default_input_modes``/``default_output_modes`` are always
    ``["text/plain"]`` (``execute()`` only extracts text via
    ``context.get_user_input()`` and only emits text via
    ``new_text_part``), and ``capabilities`` is always
    ``AgentCapabilities(streaming=False)`` (``execute()`` publishes
    only the final terminal state, no incremental updates — see the
    streaming executor variant tracked as a follow-up, issue #814).
    Everything else — pure descriptive metadata that doesn't claim
    anything about what the executor *does* — mirrors ``AgentCard``
    directly.

    Args:
        name (str): The agent's name, shown to peers.
        description (str): The agent's description, shown to peers.
        url (str): The deployment URL this agent will be served at.
            Must be supplied by the caller — nothing in this framework
            can infer where an ``LLMAgentA2AExecutor`` will actually be
            deployed.
        version (str): The agent's version string. Defaults to
            ``"0.1.0"``.
        skills (list[AgentSkill] | None): The agent's declared skills.
            Defaults to an empty list.
        provider (AgentProvider | None): The agent's provider
            organisation. Defaults to unset.
        documentation_url (str | None): URL to the agent's
            documentation. Defaults to unset.
        security_schemes (dict[str, SecurityScheme] | None): Named
            security schemes the agent supports. Defaults to none.
        security_requirements (list[SecurityRequirement] | None):
            Security requirements callers must satisfy. Defaults to
            none.
        signatures (list[AgentCardSignature] | None): Cryptographic
            signatures over the card. Defaults to none.
        icon_url (str | None): URL to an icon representing the agent.
            Defaults to unset.

    Returns:
        AgentCard: The constructed card, with a single
            ``supported_interfaces`` entry advertising the JSON-RPC
            transport at ``url``.
    """
    return AgentCard(
        # fixed: describes LLMAgentA2AExecutor's actual behavior, not
        # passed through as a parameter
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_1_0,
            ),
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        # descriptive metadata only -- mirrors AgentCard directly
        name=name,
        description=description,
        version=version,
        skills=skills or [],
        provider=provider,
        documentation_url=documentation_url,
        security_schemes=security_schemes,
        security_requirements=security_requirements,
        signatures=signatures,
        icon_url=icon_url,
    )
