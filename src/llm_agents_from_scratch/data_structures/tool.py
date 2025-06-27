"""Data Structures for Tools."""

from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    """Tool call.

    Attributes:
        tool_name: Name of tool to call.
        arguments: The arguments to pass to the tool execution.
    """

    tool_name: str
    arguments: dict[str, Any]


class ToolCallResult(BaseModel):
    """Result of a tool call execution.

    Attributes:
        content: The content of tool call.
        error: Whether or not the tool call yielded an error.
    """

    tool_call: ToolCall
    content: Any | None
    error: bool = False
