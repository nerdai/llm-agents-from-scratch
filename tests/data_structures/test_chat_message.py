from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ToolCall,
    ToolCallResult,
)


def test_tool_call_result_to_chat_message() -> None:
    """Test conversion of tool call result to an ChatMessage."""
    tool_call_result = ToolCallResult(
        tool_call=ToolCall(
            tool_name="a fake tool",
            arguments={"arg1": 1},
        ),
        content="Some content",
        error=False,
    )

    converted = ChatMessage.from_tool_call_result(tool_call_result)

    assert converted.role == "tool"
    assert converted.content == tool_call_result.model_dump_json(indent=4)
