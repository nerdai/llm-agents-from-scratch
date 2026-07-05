"""Human in the loop via HumanInputTool."""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from ...base.tool import BaseTool
from ...data_structures import ToolCall, ToolCallResult


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
        choices = tool_call.arguments.get("choices")
        try:
            console = Console()
            console.print(
                Panel(prompt, title="Human Input", border_style="yellow"),
            )
            if choices:
                response = Prompt.ask(">", choices=choices, console=console)
            else:
                response = Prompt.ask(">", console=console)
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
