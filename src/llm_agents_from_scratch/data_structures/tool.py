"""Data Structures for Tools."""

import uuid
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Tool call.

    Attributes:
        id_: String identifier for tool call.
        tool_name: Name of tool to call.
        arguments: The arguments to pass to the tool execution.
    """

    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: dict[str, Any]


class ToolCallResult(BaseModel):
    """Result of a tool call execution.

    Attributes:
        tool_call_id: The id of the associated tool call.
        content: The content of tool call.
        error: Whether or not the tool call yielded an error.
    """

    tool_call_id: str
    content: Any | None
    error: bool = False
