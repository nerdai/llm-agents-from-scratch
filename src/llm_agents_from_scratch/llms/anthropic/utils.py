"""Converters for the Anthropic Messages API."""

from typing import TYPE_CHECKING, Any, cast

from pydantic import Field, ValidationError

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


class AnthropicChatMessage(ChatMessage):
    """A ChatMessage that remembers the response blocks it was built from.

    The Messages API requires an assistant turn to be replayed exactly as
    it was received -- including `thinking` / `redacted_thinking` blocks
    and their signatures, in order -- before the tool results that answer
    it, or the follow-up request is rejected. `ChatMessage` only keeps the
    text and the tool calls, so this subclass carries the original blocks
    alongside them for `chat_message_to_anthropic_message_param` to
    replay verbatim.

    Attributes:
        raw_content (list[Any]): The `content` blocks of the
            ~anthropic.types.Message this message was converted from.
    """

    raw_content: list[Any] = Field(default_factory=list)


def anthropic_message_to_chat_message(
    message: "Message",
) -> AnthropicChatMessage:
    """Convert an ~anthropic.types.Message to an AnthropicChatMessage.

    Text blocks are concatenated into `content`; every `tool_use` block
    becomes a `ToolCall`. Every block, including the thinking and server
    tool result kinds that have no ChatMessage equivalent, is kept on
    `raw_content` so the turn can be replayed unchanged.
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
    return AnthropicChatMessage(
        role=ChatRole.ASSISTANT,
        content=content,
        tool_calls=tool_calls,
        raw_content=list(message.content),
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
            "An error occurred converting a ChatMessage to an "
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

    - an `AnthropicChatMessage` replays its original response blocks
      verbatim, so `thinking` / `redacted_thinking` blocks and their
      signatures survive the round trip the API requires
    - other assistant messages carrying tool calls become an assistant turn
      whose content is the text (if any) followed by one `tool_use` block
      per call
    - `tool` messages become a `user` turn holding a single `tool_result`
      block (see `chat_message_to_tool_result_block`)
    - everything else maps 1:1 to a plain text turn
    """
    if (
        isinstance(chat_message, AnthropicChatMessage)
        and chat_message.raw_content
    ):
        return {
            "role": "assistant",
            "content": cast(
                "list[ContentBlockParam]",
                [
                    block.model_dump(exclude_none=True)
                    for block in chat_message.raw_content
                ],
            ),
        }

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
