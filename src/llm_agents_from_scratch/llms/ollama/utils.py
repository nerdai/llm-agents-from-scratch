"""Ollama utils."""

from ollama import Message as OllamaMessage

from llm_agents_from_scratch.data_structures import ChatMessage, ChatRole


def ollama_message_to_chat_message(
    ollama_message: OllamaMessage,
) -> ChatMessage:
    """Convert an ~ollama.Message to ChatMessage.

    Args:
        ollama_message (Message): The ~ollama.Message to convert.

    Returns:
        ChatMessage: The converted message.
    """
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
    return ChatMessage(role=role, content=ollama_message.content)


def chat_message_to_ollama_message(chat_message: ChatMessage) -> OllamaMessage:
    """Convert a ChatMessage to an ~ollama.Message type.

    Args:
        chat_message (ChatMessage): The ChatMessage instance to convert.

    Returns:
        OllamaMessage: The converted message.
    """
    match chat_message.role:
        case ChatRole.ASSISTANT:
            role = "assistant"
        case ChatRole.TOOL:
            role = "tool"
        case ChatRole.USER:
            role = "user"
        case ChatRole.SYSTEM:
            role = "system"
        case _:
            msg = (
                "Failed to convert ChatMessage to ~ollama.message due to "
                f"unsupported role: `{chat_message.role}`."
            )
            raise RuntimeError(msg)

    return OllamaMessage(
        role=role,
        content=chat_message.content,
    )
