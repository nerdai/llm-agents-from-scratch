"""UseSubAgentTool — dispatches a task to a named sub-agent."""

import json
from typing import Any

from llm_agents_from_scratch.base.tool import AsyncBaseTool
from llm_agents_from_scratch.data_structures import (
    Task,
    ToolCall,
    ToolCallResult,
)

from .spec import SubAgentSpec


class UseSubAgentTool(AsyncBaseTool):
    """Async tool that dispatches a task to a registered sub-agent.

    Each sub-agent registered with the parent coordinator is callable via this
    single tool. The LLM selects a sub-agent by name (constrained to an enum
    of registered names) and provides a free-text task instruction. The tool
    runs the sub-agent to completion and returns only ``result.content``,
    keeping trajectories isolated.

    All sub-agent exceptions — including ``MaxStepsReachedError`` — are caught
    and returned as ``ToolCallResult(error=True)`` so the coordinator can
    re-plan rather than crash.

    Attributes:
        subagents (dict[str, SubAgentSpec]): Registered sub-agents, keyed by
            name.
    """

    def __init__(self, subagents: dict[str, SubAgentSpec]) -> None:
        """Initialise with a registry of sub-agents.

        Args:
            subagents (dict[str, SubAgentSpec]): Sub-agents to register, keyed
                by name.
        """
        self._subagents = subagents

    @property
    def name(self) -> str:
        """Name of the sub-agent dispatch tool."""
        return "from_scratch__use_subagent"

    @property
    def description(self) -> str:
        """Description of the sub-agent dispatch tool."""
        return (
            "Dispatch a task to a named sub-agent and return its result. "
            "Only call this tool with a sub-agent name from the "
            "<available_subagents> catalog."
        )

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters.

        The ``name`` field is constrained to an enum of registered sub-agent
        names so the LLM can only dispatch to agents that exist.
        """
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": list(self._subagents),
                    "description": (
                        "Name of the sub-agent to dispatch the task to."
                    ),
                },
                "task": {
                    "type": "string",
                    "description": "The task instruction for the sub-agent.",
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
        """Dispatch a task to the named sub-agent and return its result.

        Args:
            tool_call (ToolCall): The tool call to execute.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ToolCallResult: The sub-agent's ``result.content`` on success, or
                an error result if the sub-agent raises for any reason.
        """
        subagent_name = tool_call.arguments.get("name")
        task_instruction = tool_call.arguments.get("task")

        if not isinstance(subagent_name, str):
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
        if subagent_name not in self._subagents:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                error=True,
                content=json.dumps(
                    {
                        "error_type": "ValueError",
                        "message": (f"Sub-agent '{subagent_name}' not found."),
                    },
                ),
            )

        spec = self._subagents[subagent_name]
        try:
            task_handler = spec.agent.run(
                Task(instruction=task_instruction),
                max_steps=spec.max_steps,
            )
            result = await task_handler
        except Exception as e:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                error=True,
                content=json.dumps(
                    {
                        "error_type": type(e).__name__,
                        "subagent": subagent_name,
                        "message": str(e),
                    },
                ),
            )

        return ToolCallResult(
            tool_call_id=tool_call.id_,
            error=False,
            content=result.content,
        )
