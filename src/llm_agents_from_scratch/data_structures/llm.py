"""Data Structures for LLMs"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ChatMessage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    role: ChatRole
    content: str
    tool_calls: list[dict[str, Any]] | None = None


class CompleteResult(BaseModel):
    response: str
    full_response: str
