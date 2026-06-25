"""Human in the loop via HumanInputTool."""

from typing import Any

from ...base.tool import BaseTool
from ...data_structures import ToolCall, ToolCallResult


class HumanInputTool(BaseTool):
    """_summary_

    Args:
        BaseTool (_type_): _description_
    """

    @property
    def name(self) -> str:
        return "HumanInputTool"

    @property
    def description(self) -> str:
        return ""

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for human input parameters."""
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Prompt to provide human.",
                },
            },
        }

    def __call__(
        self,
        tool_call: ToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ToolCallResult:
        prompt = tool_call.arguments.get("prompt")
        response = input(prompt)

        return ToolCallResult(
            tool_call_id=tool_call.id_,
            content=response,
        )
