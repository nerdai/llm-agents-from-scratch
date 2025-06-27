"""Ollama utils."""

from ollama import Message as OllamaMessage
from typing_extensions import assert_never

from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ChatRole,
    ToolCall,
)


def ollama_message_to_chat_message(
    ollama_message: OllamaMessage,
) -> ChatMessage:
    """Convert an ~ollama.Message to ChatMessage.

    Args:
        ollama_message (Message): The ~ollama.Message to convert.

    Returns:
        ChatMessage: The converted message.
    """
    # role
    match ollama_message.role:
        case "assistant":
            role = ChatRole.ASSISTANT
        case "tool":
            role = ChatRole.TOOL
        case "user":
            role = ChatRole.USER
        case "system":
            role = ChatRole.SYSTEM
        case _:
            msg = (
                "Failed to convert ~ollama.Message due to invalid role: "
                f"`{ollama_message.role}`."
            )
            raise RuntimeError(msg)

    # convert tools
    converted_tool_calls = (
        [
            ToolCall(
                tool_name=o_tool_call.function.name,
                arguments=o_tool_call.function.arguments,
            )
            for o_tool_call in ollama_message.tool_calls
        ]
        if ollama_message.tool_calls
        else None
    )

    return ChatMessage(
        role=role,
        content=ollama_message.content,
        tool_calls=converted_tool_calls,
    )


def chat_message_to_ollama_message(chat_message: ChatMessage) -> OllamaMessage:
    """Convert a ChatMessage to an ~ollama.Message type.

    Args:
        chat_message (ChatMessage): The ChatMessage instance to convert.

    Returns:
        OllamaMessage: The converted message.
    """
    # role
    match chat_message.role:
        case ChatRole.ASSISTANT:
            role = "assistant"
        case ChatRole.TOOL:
            role = "tool"
        case ChatRole.USER:
            role = "user"
        case ChatRole.SYSTEM:
            role = "system"
        case _:  # pragma: no cover
            assert_never(chat_message.role)

    # convert tool calls
    converted_tool_calls = (
        [
            OllamaMessage.ToolCall(
                function=OllamaMessage.ToolCall.Function(
                    name=tc.tool_name,
                    arguments=tc.arguments,
                ),
            )
            for tc in chat_message.tool_calls
        ]
        if chat_message.tool_calls
        else None
    )

    return OllamaMessage(
        role=role,
        content=chat_message.content,
        tool_calls=converted_tool_calls,
    )
