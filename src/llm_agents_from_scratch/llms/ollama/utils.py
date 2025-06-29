"""Ollama utils."""

from typing import Any

from ollama import Message as OllamaMessage
from ollama import Tool as OllamaTool
from typing_extensions import assert_never

from llm_agents_from_scratch.base.tool import AsyncBaseTool, BaseTool
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ChatRole,
    ToolCall,
    ToolCallResult,
)

DEFAULT_TOOL_RESPONSE_TEMPLATE = """
The below is a tool call response for a given tool call.
<tool-call>
tool name: {tool_name}
arguments: {arguments}
</tool-call>

<result>
{tool_call_result}
</result>
"""


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


def tool_call_result_to_ollama_message(
    tool_call_result: ToolCallResult,
) -> OllamaMessage:
    """Convert a tool call result to an ~ollama.Message.

    Args:
        tool_call_result (ToolCallResult): The tool call result.

    Returns:
        OllamaMessage: The converted message.
    """
    formatted_content = DEFAULT_TOOL_RESPONSE_TEMPLATE.format(
        tool_name=tool_call_result.tool_call.tool_name,
        arguments=tool_call_result.tool_call.arguments,
        tool_call_result=tool_call_result.content,
    )

    return OllamaMessage(
        role="tool",
        content=formatted_content,
    )


def get_tool_json_schema(tool: BaseTool | AsyncBaseTool) -> dict[str, Any]:
    """Prepare a tool as a JSON schema.

    Args:
        tool (BaseTool | AsyncBaseTool): The tool for which to get the JSON
            schema.

    Returns:
        dict[str, Any]: The JSON schema for the tool.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_json_schema,
        },
    }


def tool_to_ollama_tool(tool: BaseTool | AsyncBaseTool) -> OllamaTool:
    """Convert a BaseTool or AsyncBaseTool to an ~ollama.Tool type.

    Args:
        tool (BaseTool | AsyncBaseTool): The base tool to convert.

    Returns:
        ~ollama.Tool: The converted tool.
    """
    json_schema = get_tool_json_schema(tool)
    return OllamaTool.model_validate(json_schema)
