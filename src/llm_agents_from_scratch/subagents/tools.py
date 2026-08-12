"""UseSubAgentTool — dispatches a task to a named subagent."""

import json
from typing import Any

from llm_agents_from_scratch.base.tool import AsyncBaseTool
from llm_agents_from_scratch.data_structures import (
    Task,
    ToolCall,
    ToolCallResult,
)
from llm_agents_from_scratch.errors import SubAgentNotFoundError
from llm_agents_from_scratch.logger import current_subagent_name

from .spec import SubAgentSpec


class UseSubAgentTool(AsyncBaseTool):
    """Async tool that dispatches a task to a registered subagent.

    Each subagent registered with the parent coordinator is callable via this
    single tool. The LLM selects a subagent by name (constrained to an enum
    of registered names) and provides a free-text task instruction. The tool
    runs the subagent to completion and returns only ``result.content``,
    keeping trajectories isolated.

    All subagent exceptions — including ``MaxStepsReachedError`` — are caught
    and returned as ``ToolCallResult(error=True)`` so the coordinator can
    re-plan rather than crash.

    Attributes:
        subagents_registry (dict[str, SubAgentSpec]): Registered subagents,
            keyed by name.
    """

    def __init__(self, subagents_registry: dict[str, SubAgentSpec]) -> None:
        """Initialise with a registry of subagents.

        Args:
            subagents_registry (dict[str, SubAgentSpec]): subagents to
                register, keyed by name.
        """
        self._subagents_registry = subagents_registry

    @property
    def name(self) -> str:
        """Name of the subagent dispatch tool."""
        return "from_scratch__use_subagent"

    @property
    def description(self) -> str:
        """Description of the subagent dispatch tool."""
        return (
            "Dispatch a task to a named subagent and return its result. "
            "Only call this tool with a subagent name from the "
            "<available_subagents> catalog."
        )

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters.

        The ``name`` field is constrained to an enum of registered subagent
        names so the LLM can only dispatch to agents that exist.
        """
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": sorted(self._subagents_registry),
                    "description": (
                        "Name of the subagent to dispatch the task to."
                    ),
                },
                "task": {
                    "type": "string",
                    "description": "The task instruction for the subagent.",
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
        """Dispatch a task to the named subagent and return its result.

        Builds a fresh ``LLMAgent`` from ``spec.builder`` on every call —
        the spec holds a recipe, not a live agent. Whatever the builder
        was given directly (memory stores, MCP providers) is reused
        as-is; only the agent shell is rebuilt.

        Sets ``current_subagent_name`` before calling ``agent.run()`` so
        the ``asyncio.Task`` it creates copies a context with the name
        already set, and resets it once the dispatch settles. This is the
        only call site that ever sets ``current_subagent_name``.

        Safe under normal usage: ``run_step()`` always reaches this method
        through ``asyncio.gather`` (even for a single tool call), so the
        context mutated here is already a fork of the coordinator's own
        context, never the coordinator's context itself — concurrent or
        nested dispatches can't collide, since each fork owns an
        independent copy from the moment it's created, before any
        ``set()`` runs. Calling this tool directly, bypassing
        ``run_step()`` (e.g. for a manual dispatch demo), is the one path
        where ``set()`` lands on whatever context the caller happens to
        be in — still safe as long as nothing else concurrently shares
        that context, which the ``try``/``finally`` reset below
        guarantees for this call in isolation.

        Args:
            tool_call (ToolCall): The tool call to execute.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ToolCallResult: The subagent's ``result.content`` on success, or
                an error result if the subagent raises for any reason.
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
        spec = self._subagents_registry.get(subagent_name)
        token = current_subagent_name.set(subagent_name)
        try:
            if spec is None:
                raise SubAgentNotFoundError(
                    f"subagent '{subagent_name}' not found.",
                )
            agent = await spec.builder.build()
            result = await agent.run(
                Task(instruction=task_instruction),
                max_steps=spec.max_steps,
                skills_scopes=spec.skills_scopes,
                explicit_only_skills=spec.explicit_only_skills,
            )
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
        finally:
            current_subagent_name.reset(token)

        return ToolCallResult(
            tool_call_id=tool_call.id_,
            error=False,
            content=result.content,
        )
