"""Data Structures for LLMs."""

from enum import Enum

from pydantic import BaseModel, ConfigDict

from .tool import ToolCall


class ChatRole(str, Enum):
    """Roles for chat messages."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """The chat message data model.

    Attributes:
        role: The role of the message.
        content: The content of the message.
        tool_calls: Tool calls associated with the message.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    role: ChatRole
    content: str
    tool_calls: list[ToolCall] | None = None


class CompleteResult(BaseModel):
    """The llm completion result data model.

    Attributes:
        response: The completion response provided by the LLM.
        full_response: Input prompt and completion text.
    """

    response: str
    prompt: str
