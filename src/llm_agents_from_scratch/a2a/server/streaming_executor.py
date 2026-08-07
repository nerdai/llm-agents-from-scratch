"""StreamingLLMAgentA2AExecutor — streaming variant of LLMAgentA2AExecutor."""

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
from llm_agents_from_scratch.data_structures import Task, TaskResult, TaskStep


class StreamingLLMAgentA2AExecutor(AgentExecutor):
    """Bridges inbound A2A tasks to an ``LLMAgent``, streaming updates.

    Drives ``LLMAgent.run_supervised()`` directly instead of ``run()``:
    ``SupervisedTaskHandler`` has no background task at all — execution
    is entirely caller-driven via ``get_next_step()``/``run_step()``
    (built for Ch8's HITL use case, structurally identical to what
    streaming needs). This executor calls that loop itself, publishing
    a ``TaskStatusUpdateEvent`` after each step instead of prompting a
    human — the same substitution ``LLMAgentA2AExecutor`` makes for the
    non-streaming case, just at per-step instead of whole-run
    granularity.

    Two updates are published per step, not one: ``get_next_step()``
    makes its own LLM call to decide routing (except on the very first
    step) — real, awaitable work in its own right, not just
    bookkeeping — so its result (the planned instruction) is published
    before ``run_step()`` executes it, and that step's result is
    published after.

    Because there's no separate background task, cancellation here is
    simpler than ``LLMAgentA2AExecutor``'s: the SDK's own producer loop
    already runs *this* executor's ``execute()`` as its own task, so
    cancelling that task interrupts whatever step is currently
    in-flight directly — there's no orphaned worker left running
    underneath, unlike ``LLMAgentA2AExecutor.cancel()``'s need to
    separately cancel ``TaskHandler.background_task``.

    Attributes:
        agent (LLMAgent): The agent this executor serves.
        _task_handlers (dict[str, LLMAgent.SupervisedTaskHandler]):
            In-flight runs, keyed by task_id. ``execute()`` and
            ``cancel()`` are separate calls with no shared local state
            connecting them — this registry is what lets ``cancel()``
            find the specific ``SupervisedTaskHandler`` a concurrent
            ``execute()`` call (for a different task_id) is driving.
    """

    def __init__(self, agent: LLMAgent) -> None:
        """Initialise with the agent to serve.

        Args:
            agent (LLMAgent): The agent to bridge inbound A2A tasks to.
        """
        self.agent = agent
        self._task_handlers: dict[str, LLMAgent.SupervisedTaskHandler] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Runs the agent step by step, publishing an update per step.

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
        task_handler = await self.agent.run_supervised(
            Task(instruction=instruction),
        )
        self._task_handlers[task.id] = task_handler
        try:
            step_result = None
            while not task_handler.done():
                next_step = await task_handler.get_next_step(step_result)
                match next_step:
                    case TaskStep():
                        await updater.update_status(
                            TaskState.TASK_STATE_WORKING,
                            message=updater.new_agent_message(
                                [new_text_part(next_step.instruction)],
                            ),
                        )
                        step_result = await task_handler.run_step(next_step)
                        await updater.update_status(
                            TaskState.TASK_STATE_WORKING,
                            message=updater.new_agent_message(
                                [new_text_part(step_result.content)],
                            ),
                        )
                    case TaskResult():
                        await task_handler.complete(next_step)
        except asyncio.CancelledError:
            # Not caught by except Exception below (CancelledError is a
            # BaseException, not an Exception) -- this clause is purely
            # explicit documentation of that fact, not a behavior
            # change: it must propagate uncaught so the SDK's own
            # producer-loop cleanup (which closes event_queue and lets
            # the producer task actually terminate) still runs. The
            # finally block below still settles _task_handlers either
            # way.
            raise
        except Exception as e:
            await updater.update_status(
                TaskState.TASK_STATE_FAILED,
                message=updater.new_agent_message([new_text_part(str(e))]),
            )
            return
        finally:
            self._task_handlers.pop(task.id, None)

        result = task_handler.result()
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

        # No background_task to worry about here (SupervisedTaskHandler
        # has none) -- the SDK already cancels the producer task
        # running our own execute() before calling cancel(), which
        # interrupts whatever step is currently in-flight directly.
        # Publish CANCELED first regardless, for the same reason as
        # LLMAgentA2AExecutor.cancel(): avoid racing the SDK's own
        # producer-loop cleanup over this event_queue.
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()

        if not task_handler.done():
            task_handler.cancel()


def build_streaming_agent_card(  # noqa: PLR0913, PLR0917
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
    """Builds an ``AgentCard`` for serving a ``StreamingLLMAgentA2AExecutor``.

    Mirrors ``executor.build_agent_card()`` exactly, except
    ``capabilities`` is ``AgentCapabilities(streaming=True)`` — this
    executor genuinely publishes incremental status updates per step,
    unlike the non-streaming ``LLMAgentA2AExecutor``. As with
    ``build_agent_card()``, every field describing executor *behavior*
    is fixed rather than exposed as a parameter (``supported_interfaces``,
    the default I/O modes, ``capabilities``); pure descriptive metadata
    that claims nothing about behavior stays overridable.

    Args:
        name (str): The agent's name, shown to peers.
        description (str): The agent's description, shown to peers.
        url (str): The deployment URL this agent will be served at.
            Must be supplied by the caller — nothing in this framework
            can infer where a ``StreamingLLMAgentA2AExecutor`` will
            actually be deployed.
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
            transport at ``url``, and streaming enabled.
    """
    return AgentCard(
        # fixed: describes StreamingLLMAgentA2AExecutor's actual
        # behavior, not passed through as a parameter
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_1_0,
            ),
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
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
