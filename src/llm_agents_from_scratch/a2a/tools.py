"""UseA2AAgentTool — dispatches a task to a named A2A peer agent."""

import json
from typing import Any

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role as A2ARole
from a2a.types import SendMessageRequest, StreamResponse

from llm_agents_from_scratch.base.tool import AsyncBaseTool
from llm_agents_from_scratch.data_structures import ToolCall, ToolCallResult
from llm_agents_from_scratch.errors import A2AAgentNotFoundError

from .spec import A2AAgentSpec
from .utils import build_result


class UseA2AAgentTool(AsyncBaseTool):
    """Async tool that dispatches a task to a registered A2A peer agent.

    Each A2A peer registered with the coordinator is callable via this
    single tool — mirrors ``UseSubAgentTool``'s shape (one generic
    dispatch tool over a registry, ``name`` constrained to an enum), for
    the same reason: A2A peers are homogeneous (every one takes a text
    instruction and returns an artifact), unlike MCP tools which are
    heterogeneous and so get one tool each.

    The clean diff from ``UseSubAgentTool``: an optional ``task_id`` to
    resume a remote task a peer previously parked in
    ``TASK_STATE_INPUT_REQUIRED``. There is no local state tracking
    pending peer tasks — the coordinator holds the task id (echoed back
    via ``A2A_INPUT_REQUIRED_TEMPLATE``) and passes it back explicitly,
    keeping this tool stateless over the registry just like
    ``UseSubAgentTool`` is.

    Stateless over the SDK client too: ``A2AAgentSpec`` never holds a
    live client (see #783), so this tool calls ``create_client()`` fresh
    on every dispatch and always closes it before returning.

    All exceptions — connection errors, an unknown ``task_id``, etc. —
    are caught and returned as ``ToolCallResult(error=True)`` so the
    coordinator can re-plan rather than crash. An explicit
    ``TASK_STATE_FAILED``/``TASK_STATE_CANCELED``/``TASK_STATE_REJECTED``
    result from the peer is mapped into the same error JSON shape.

    Attributes:
        a2a_agents_registry (dict[str, A2AAgentSpec]): Registered A2A
            peers, keyed by name.
    """

    def __init__(self, a2a_agents_registry: dict[str, A2AAgentSpec]) -> None:
        """Initialise with a registry of A2A peer agents.

        Args:
            a2a_agents_registry (dict[str, A2AAgentSpec]): A2A peers to
                register, keyed by name.
        """
        self._a2a_agents_registry = a2a_agents_registry

    @property
    def name(self) -> str:
        """Name of the A2A peer dispatch tool."""
        return "from_scratch__use_a2a_agent"

    @property
    def description(self) -> str:
        """Description of the A2A peer dispatch tool."""
        return (
            "Dispatch a task to a named A2A peer agent and return its "
            "result. Only call this tool with a peer name from the "
            "<available_a2a_agents> catalog."
        )

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters.

        The ``name`` field is constrained to an enum of registered A2A
        peer names so the LLM can only dispatch to peers that exist.
        ``task_id`` is optional — set it only to resume a remote task a
        peer previously parked in ``TASK_STATE_INPUT_REQUIRED``.
        """
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": sorted(self._a2a_agents_registry),
                    "description": (
                        "Name of the A2A peer agent to dispatch the task to."
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "The task instruction for the peer agent, or "
                        "the requested information if resuming a task "
                        "that returned input_required."
                    ),
                },
                "task_id": {
                    "type": "string",
                    "description": (
                        "Optional remote task id to resume, echoed back "
                        "from a prior input_required result. Omit to "
                        "start a new task."
                    ),
                },
            },
            "required": ["name", "task"],
        }

    async def __call__(
        self,
        tool_call: ToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ToolCallResult:
        """Dispatch a task to the named A2A peer and return its result.

        Args:
            tool_call (ToolCall): The tool call to execute.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ToolCallResult: The peer's result content on success, an
                ``A2A_INPUT_REQUIRED_TEMPLATE``-wrapped result if the
                peer needs more information, or an error result if
                dispatch fails for any reason.
        """
        agent_name = tool_call.arguments.get("name")
        task_instruction = tool_call.arguments.get("task")
        task_id = tool_call.arguments.get("task_id")

        if not isinstance(agent_name, str):
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                error=True,
                content=json.dumps(
                    {
                        "error_type": "ValueError",
                        "message": "'name' argument must be a string.",
                    },
                ),
            )
        if not isinstance(task_instruction, str):
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                error=True,
                content=json.dumps(
                    {
                        "error_type": "ValueError",
                        "message": "'task' argument must be a string.",
                    },
                ),
            )
        if task_id is not None and not isinstance(task_id, str):
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                error=True,
                content=json.dumps(
                    {
                        "error_type": "ValueError",
                        "message": "'task_id' argument must be a string.",
                    },
                ),
            )

        spec = self._a2a_agents_registry.get(agent_name)
        if spec is None:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                error=True,
                content=json.dumps(
                    {
                        "error_type": A2AAgentNotFoundError.__name__,
                        "a2a_agent": agent_name,
                        "message": f"A2A agent '{agent_name}' not found.",
                    },
                ),
            )

        # The context manager guarantees httpx_client is closed even if
        # create_client() itself raises. client.close() is still called
        # separately below for transport-level cleanup beyond httpx —
        # e.g. the gRPC transport's close() releases its own channel,
        # not httpx_client. A harmless double-close of httpx_client can
        # happen on transports that do route through it (aclose() is
        # idempotent).
        async with httpx.AsyncClient(headers=spec.headers) as httpx_client:
            client = None
            try:
                client = await create_client(
                    agent=spec.agent_card,
                    client_config=ClientConfig(
                        streaming=False,
                        httpx_client=httpx_client,
                    ),
                )
                message = new_text_message(
                    text=task_instruction,
                    task_id=task_id,
                    role=A2ARole.ROLE_USER,
                )
                request = SendMessageRequest(message=message)

                response: StreamResponse | None = None
                async for chunk in client.send_message(request):
                    response = chunk
            except Exception as e:
                return ToolCallResult(
                    tool_call_id=tool_call.id_,
                    error=True,
                    content=json.dumps(
                        {
                            "error_type": type(e).__name__,
                            "a2a_agent": agent_name,
                            "message": str(e),
                        },
                    ),
                )
            finally:
                # Swallow close-time errors so they can't override a
                # result already computed above, keeping the "all
                # exceptions are caught" guarantee true during cleanup.
                if client is not None:
                    try:  # noqa: SIM105
                        await client.close()
                    except Exception:
                        pass

        return build_result(response, agent_name, tool_call.id_)
