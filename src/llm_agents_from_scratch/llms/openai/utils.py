"""Converters for OpenAI Responses API."""

import json
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
    from openai.types.responses import (
        Response,
        ResponseInputItemParam,
        ToolParam,
    )


def openai_response_to_chat_message(openai_response: "Response") -> ChatMessage:
    """Convert an ~openai.Response to ChatMessage."""
    content = openai_response.output_text
    role = ChatRole.ASSISTANT

    # collect tool calls
    tool_calls = []
    for item in openai_response.output:
        if item.type != "function_call":
            continue

        tool_calls.append(
            ToolCall(
                id_=item.call_id,
                tool_name=item.name,
                arguments=json.loads(item.arguments),
            ),
        )

    return ChatMessage(content=content, role=role, tool_calls=tool_calls)


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
        try:
            tool_call_result = ToolCallResult.model_validate_json(
                chat_message.content,
            )
        except ValidationError as e:
            msg = (
                "An error occured converting a ChatMessage "
                "to an openai.ResponseInputParam. "
                f"Unable to build ToolCallResult from ChatMessage: {str(e)}",
            )
            raise DataConversionError(msg) from e
        function_call_output: FunctionCallOutput = {
            "type": "function_call_output",
            "call_id": tool_call_result.tool_call_id,
            "output": tool_call_result.model_dump_json(
                exclude="tool_call_id",
                indent=2,
            ),
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
        "parameters": tool.parameters_json_schema,
        "strict": True,
    }
    return openai_tool
