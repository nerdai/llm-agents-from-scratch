"""Human in the loop via HumanInputTool."""

from typing import Any

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
        return "human_input"

    @property
    def description(self) -> str:
        """Description of the human-input tool."""
        return (
            "Ask the human operator a question and wait for their response. "
            "Use this tool when you need clarification or additional "
            "information that is not available from the task context."
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

        Blocks until the operator submits a response via stdin.

        Args:
            tool_call (ToolCall): The tool call containing the ``prompt``
                argument.
            *args (Any): Additional positional arguments (unused).
            **kwargs (Any): Additional keyword arguments (unused).

        Returns:
            ToolCallResult: The human's response as the content.
        """
        prompt = tool_call.arguments.get("prompt", "")
        response = input(prompt)

        return ToolCallResult(
            tool_call_id=tool_call.id_,
            content=response,
        )
