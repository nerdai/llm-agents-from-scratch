"""Converters for the Anthropic Messages API."""

from typing import TYPE_CHECKING

from pydantic import ValidationError

from llm_agents_from_scratch.base.tool import Tool
from llm_agents_from_scratch.data_structures.llm import ChatMessage, ChatRole
from llm_agents_from_scratch.data_structures.tool import (
    ToolCall,
    ToolCallResult,
)

from .errors import DataConversionError

if TYPE_CHECKING:  # pragma: no cover
    from anthropic.types import (
        ContentBlockParam,
        Message,
        MessageParam,
        ToolParam,
        ToolResultBlockParam,
    )


def anthropic_message_to_chat_message(message: "Message") -> ChatMessage:
    """Convert an ~anthropic.types.Message to ChatMessage.

    Text blocks are concatenated into `content`; every `tool_use` block
    becomes a `ToolCall`. Other block types (thinking, server tool
    results, ...) are ignored.
    """
    content = "".join(
        block.text for block in message.content if block.type == "text"
    )
    tool_calls = [
        ToolCall(
            id_=block.id,
            tool_name=block.name,
            arguments=block.input if isinstance(block.input, dict) else {},
        )
        for block in message.content
        if block.type == "tool_use"
    ]
    return ChatMessage(
        role=ChatRole.ASSISTANT,
        content=content,
        tool_calls=tool_calls,
    )


def chat_message_to_tool_result_block(
    chat_message: ChatMessage,
) -> "ToolResultBlockParam":
    """Convert a `tool` ChatMessage to an ~anthropic.types.ToolResultBlockParam.

    The Messages API has no tool role: results travel as `tool_result`
    content blocks inside a `user` turn, one block per call.

    Raises:
        DataConversionError: If the message's content is not a serialized
            `ToolCallResult`.
    """
    try:
        tool_call_result = ToolCallResult.model_validate_json(
            chat_message.content,
        )
    except ValidationError as e:
        msg = (
            "An error occured converting a ChatMessage to an "
            "anthropic.ToolResultBlockParam. Unable to build ToolCallResult "
            f"from ChatMessage: {str(e)}"
        )
        raise DataConversionError(msg) from e
    return {
        "type": "tool_result",
        "tool_use_id": tool_call_result.tool_call_id,
        "content": tool_call_result.model_dump_json(
            exclude={"tool_call_id"},
            indent=2,
        ),
        "is_error": tool_call_result.error,
    }


def chat_message_to_anthropic_message_param(
    chat_message: ChatMessage,
) -> "MessageParam":
    """Convert a ChatMessage to an ~anthropic.types.MessageParam.

    - assistant messages carrying tool calls become an assistant turn whose
      content is the text (if any) followed by one `tool_use` block per call
    - `tool` messages become a `user` turn holding a single `tool_result`
      block (see `chat_message_to_tool_result_block`)
    - everything else maps 1:1 to a plain text turn
    """
    if chat_message.tool_calls:
        content: list["ContentBlockParam"] = []
        if chat_message.content:
            content.append({"type": "text", "text": chat_message.content})
        content.extend(
            {
                "type": "tool_use",
                "id": tool_call.id_,
                "name": tool_call.tool_name,
                "input": tool_call.arguments,
            }
            for tool_call in chat_message.tool_calls
        )
        return {"role": "assistant", "content": content}

    if chat_message.role == ChatRole.TOOL:
        return {
            "role": "user",
            "content": [chat_message_to_tool_result_block(chat_message)],
        }

    return {
        "role": (
            "assistant" if chat_message.role == ChatRole.ASSISTANT else "user"
        ),
        "content": chat_message.content,
    }


def tool_to_anthropic_tool(tool: Tool) -> "ToolParam":
    """Convert a BaseTool or AsyncBaseTool to an ~anthropic.types.ToolParam.

    Args:
        tool (Tool): The base tool to convert.

    Returns:
        ~anthropic.types.ToolParam: The converted tool.
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters_json_schema,
    }
