"""UseA2AAgentTool — dispatches a task to a named A2A peer agent."""

import json
from typing import Any

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest, Task, TaskState

from llm_agents_from_scratch.base.tool import AsyncBaseTool
from llm_agents_from_scratch.data_structures import ToolCall, ToolCallResult
from llm_agents_from_scratch.errors import A2AAgentNotFoundError

from .constants import A2A_INPUT_REQUIRED_TEMPLATE
from .spec import A2AAgentSpec
from .utils import parts_text, task_content


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
        client = None
        try:
            if spec is None:
                raise A2AAgentNotFoundError(
                    f"A2A agent '{agent_name}' not found.",
                )

            httpx_client = httpx.AsyncClient(headers=spec.headers)
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
                role=Role.ROLE_USER,
            )
            request = SendMessageRequest(message=message)

            response = None
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
            if client is not None:
                await client.close()

        return self._build_result(response, agent_name, tool_call.id_)

    def _build_result(
        self,
        response: Any,
        agent_name: str,
        tool_call_id: str,
    ) -> ToolCallResult:
        """Turn the final ``StreamResponse`` chunk into a ``ToolCallResult``.

        Args:
            response: The last chunk yielded by ``client.send_message()``,
                or ``None`` if the peer yielded nothing.
            agent_name: The dispatched-to peer's registry name, for error
                JSON and the input-required template.
            tool_call_id: The originating tool call's id.

        Returns:
            ToolCallResult: Success with the peer's content, an
                ``A2A_INPUT_REQUIRED_TEMPLATE``-wrapped result, or an
                error result.
        """
        if response is None:
            return ToolCallResult(
                tool_call_id=tool_call_id,
                error=True,
                content=json.dumps(
                    {
                        "error_type": "A2AEmptyResponseError",
                        "a2a_agent": agent_name,
                        "message": "Peer returned no task or message.",
                    },
                ),
            )

        kind = response.WhichOneof("payload")

        if kind == "message":
            return ToolCallResult(
                tool_call_id=tool_call_id,
                error=False,
                content=parts_text(response.message.parts),
            )

        if kind == "task":
            return self._build_task_result(
                response.task,
                agent_name,
                tool_call_id,
            )

        return ToolCallResult(
            tool_call_id=tool_call_id,
            error=True,
            content=json.dumps(
                {
                    "error_type": "A2AEmptyResponseError",
                    "a2a_agent": agent_name,
                    "message": "Peer returned no task or message.",
                },
            ),
        )

    def _build_task_result(
        self,
        task: Task,
        agent_name: str,
        tool_call_id: str,
    ) -> ToolCallResult:
        """Turn a peer's final ``Task`` into a ``ToolCallResult``.

        Args:
            task: The peer's final ``Task`` state.
            agent_name: The dispatched-to peer's registry name, for error
                JSON and the input-required template.
            tool_call_id: The originating tool call's id.

        Returns:
            ToolCallResult: Success with the task's artifact content, an
                ``A2A_INPUT_REQUIRED_TEMPLATE``-wrapped result, or an
                error result for any other terminal state.
        """
        state = TaskState.Name(task.status.state)

        if state == "TASK_STATE_COMPLETED":
            return ToolCallResult(
                tool_call_id=tool_call_id,
                error=False,
                content=task_content(task),
            )

        if state == "TASK_STATE_INPUT_REQUIRED":
            question = parts_text(task.status.message.parts) or (
                task_content(task)
            )
            return ToolCallResult(
                tool_call_id=tool_call_id,
                error=False,
                content=A2A_INPUT_REQUIRED_TEMPLATE.format(
                    name=agent_name,
                    question=question,
                    task_id=task.id,
                ),
            )

        # any other state (FAILED, CANCELED, REJECTED, AUTH_REQUIRED, or
        # an unexpected non-terminal state from a non-streaming response)
        # is an error the coordinator should re-plan around
        message_text = parts_text(task.status.message.parts) or (
            f"Task ended in state {state}."
        )
        return ToolCallResult(
            tool_call_id=tool_call_id,
            error=True,
            content=json.dumps(
                {
                    "error_type": state,
                    "a2a_agent": agent_name,
                    "message": message_text,
                },
            ),
        )
