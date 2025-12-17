"""Converters for OpenAI Responses API."""

from typing import TYPE_CHECKING

from llm_agents_from_scratch.base.tool import Tool
from llm_agents_from_scratch.data_structures.llm import ChatMessage

if TYPE_CHECKING:  # pragma: no cover
    from openai.types.responses import (
        Response,
        ResponseInputItemParam,
        ToolParam,
    )


def openai_response_to_chat_message(openai_response: "Response") -> ChatMessage:
    """Convert an ~openai.Response to ChatMessage."""
    pass


def chat_message_to_openai_response_input_param(
    chat_message: ChatMessage,
) -> "ResponseInputItemParam":
    """Convert a ChatMessage to an ~openai.ResponseInputParam.

    NOTE: ResponseInputParam is a union type. This method returns one of two of
    its available options—EasyInputMessageParam or FunctionCallOutput.
    """
    from openai.types.responses.response_input_param import (  # noqa: PLC0415
        EasyInputMessageParam,
        FunctionCallOutput,
    )

    if chat_message.role == "tool":
        function_call_output: FunctionCallOutput = {
            "type": "function_call_output",
            "call_id": chat_message.tool_calls[0].id_,
            "output": str(chat_message.content),
        }
        return function_call_output

    input_message: EasyInputMessageParam = {
        "type": "message",
        "content": chat_message.content,
        "role": chat_message.role.value,
    }
    return input_message


def tool_to_openai_tool(tool: Tool) -> "ToolParam":
    """Convert a BaseTool or AsyncBaseTool to an ~openai.ToolParam type.

    Args:
        tool (Tool): The base tool to convert.

    Returns:
        ~openai.ToolParam: The converted tool.
    """
    from openai.types.responses import FunctionToolParam  # noqa: PLC0415

    openai_tool: FunctionToolParam = {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters_json_schema(),
        "strict": True,
    }
    return openai_tool
