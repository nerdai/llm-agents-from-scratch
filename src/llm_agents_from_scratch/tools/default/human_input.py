"""Human in the loop via HumanInputTool and SharedConsoleHumanInputTool."""

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from ...base.tool import AsyncBaseTool, BaseTool
from ...data_structures import ToolCall, ToolCallResult


def _prompt_human(
    prompt: str,
    choices: list[str] | None,
    agent_name: str | None = None,
) -> str:
    """Render a human-input panel and return the operator's response.

    Raises:
        EOFError: If stdin is closed.
        KeyboardInterrupt: If the operator interrupts.
    """
    console = Console()
    title = f"Human Input — {agent_name}" if agent_name else "Human Input"
    console.print(Panel(prompt, title=title, border_style="yellow"))
    if choices:
        return Prompt.ask(">", choices=choices, console=console)
    return Prompt.ask(">", console=console)


class HumanInputTool(BaseTool):
    """A tool that pauses execution to collect input from a human operator.

    The LLM calls this tool when it needs clarification, a decision, or
    additional information that cannot be inferred from the task alone.
    Execution blocks until the human responds.
    """

    @property
    def name(self) -> str:
        """Name of the human-input tool."""
        return "from_scratch__human_input"

    @property
    def description(self) -> str:
        """Description of the human-input tool."""
        return (
            "Ask the human operator a question and wait for their response. "
            "Use this tool when you need clarification or additional "
            "information that is not available from the task context. "
            "Optionally provide a list of choices to constrain the response."
        )

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for human input parameters."""
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The question or prompt to present to the human."
                    ),
                },
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of valid responses. When provided, "
                        "the human is re-prompted until they enter one of "
                        "the listed values."
                    ),
                },
            },
            "required": ["prompt"],
        }

    def __call__(
        self,
        tool_call: ToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ToolCallResult:
        """Display a prompt to the human operator and return their response.

        Blocks until the operator submits a valid response via stdin. When
        ``choices`` is provided, re-prompts until the response matches one
        of the allowed values.

        Args:
            tool_call (ToolCall): The tool call containing the ``prompt``
                argument and optional ``choices`` list.
            *args (Any): Additional positional arguments (unused).
            **kwargs (Any): Additional keyword arguments (unused).

        Returns:
            ToolCallResult: The human's response as the content.
        """
        prompt = tool_call.arguments.get("prompt", "")
        if not prompt:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content="No prompt provided.",
                error=True,
            )
        choices: list[str] | None = tool_call.arguments.get("choices")
        try:
            response = _prompt_human(prompt, choices)
        except EOFError:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content="No input received (stdin closed).",
                error=True,
            )
        except KeyboardInterrupt:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content="Human declined to provide input (KeyboardInterrupt).",
                error=True,
            )
        return ToolCallResult(
            tool_call_id=tool_call.id_,
            content=response,
        )


class SharedConsoleHumanInputTool(AsyncBaseTool):
    """Async human-input tool safe under concurrent tool execution.

    Drop-in async replacement for ``HumanInputTool`` for use in
    multi-agent systems where ``asyncio.gather`` may run tools
    concurrently. The class-level lock serializes all prompts to a
    single stdin so concurrent agents take turns rather than racing.

    The optional ``agent_name`` label is rendered in the panel title so
    the operator knows which sub-agent is asking. Typically set to
    ``SubAgentSpec.name`` at construction.

    The class-level ``_console_lock`` is shared by all instances, so
    concurrent agents take turns at a single stdin rather than racing.
    ``asyncio.Lock`` binding behaviour differs by Python version:

    - ``<=3.9``: ``__init__`` called ``get_event_loop()`` at
      construction, binding the lock to the import-time loop.
      Class-level creation was unsafe.
    - ``3.10-3.11``: ``_LoopBoundMixin`` defers binding to the first
      ``acquire()``. Construction is safe; the lock binds on first use
      and then rejects any other event loop. Tests that spin up a
      fresh loop per function must reset this attribute to a new
      ``asyncio.Lock()`` in an autouse fixture so each test gets an
      unbound lock.
    - ``3.12+``: ``_LoopBoundMixin`` removed; every ``acquire()`` calls
      ``get_running_loop()`` fresh, so there is no caching and no
      cross-loop errors.

    This project requires ``>=3.10``, so class-level creation is safe
    in production (all agents run as ``create_task()`` coroutines
    within a single ``asyncio.run()``). See
    ``tests/tools/test_default.py::reset_console_lock`` for the
    autouse fixture that handles 3.10-3.11 test isolation.

    Attributes:
        agent_name (str | None): Label shown in the panel title, e.g.
            ``"Human Input — researcher"``. ``None`` renders the
            default ``"Human Input"`` title.
    """

    # Class-level lock — all instances share one stdin, so one lock
    # suffices. See the class docstring for version-specific binding
    # behaviour.
    _console_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, agent_name: str | None = None) -> None:
        """Initialise with an optional agent-name label.

        Args:
            agent_name (str | None): Sub-agent name rendered in the
                panel title. Defaults to None.
        """
        self.agent_name = agent_name

    @property
    def name(self) -> str:
        """Name of the shared-console human-input tool."""
        return "from_scratch__human_input"

    @property
    def description(self) -> str:
        """Description of the shared-console human-input tool."""
        return (
            "Ask the human operator a question and wait for their response. "
            "Use this tool when you need clarification or additional "
            "information that is not available from the task context. "
            "Optionally provide a list of choices to constrain the response."
        )

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for human input parameters."""
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "The question or prompt to present to the human."
                    ),
                },
                "choices": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of valid responses. When provided, "
                        "the human is re-prompted until they enter one of "
                        "the listed values."
                    ),
                },
            },
            "required": ["prompt"],
        }

    async def __call__(
        self,
        tool_call: ToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ToolCallResult:
        """Acquire the console lock, then prompt the human in a thread.

        Serializes concurrent prompts via ``_console_lock`` and runs
        ``_prompt_human`` in a worker thread so the event loop stays free
        while waiting for stdin.

        Args:
            tool_call (ToolCall): The tool call containing the ``prompt``
                argument and optional ``choices`` list.
            *args (Any): Additional positional arguments (unused).
            **kwargs (Any): Additional keyword arguments (unused).

        Returns:
            ToolCallResult: The human's response as the content.
        """
        prompt = tool_call.arguments.get("prompt", "")
        if not prompt:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content="No prompt provided.",
                error=True,
            )
        choices: list[str] | None = tool_call.arguments.get("choices")
        try:
            async with SharedConsoleHumanInputTool._console_lock:
                response = await asyncio.to_thread(
                    _prompt_human,
                    prompt,
                    choices,
                    self.agent_name,
                )
        except EOFError:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content="No input received (stdin closed).",
                error=True,
            )
        except KeyboardInterrupt:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content="Human declined to provide input (KeyboardInterrupt).",
                error=True,
            )
        return ToolCallResult(
            tool_call_id=tool_call.id_,
            content=response,
        )
