from unittest.mock import MagicMock, patch

from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ToolCall,
    ToolCallResult,
)


@patch("llm_agents_from_scratch.data_structures.tool.uuid")
def test_tool_call_result_to_chat_message(mock_uuid: MagicMock) -> None:
    """Test conversion of tool call result to an ChatMessage."""
    mock_uuid.uuid4.return_value = "111"
    tool_call = ToolCall(
        tool_name="a fake tool",
        arguments={"arg1": 1},
    )
    tool_call_result = ToolCallResult(
        tool_call_id=tool_call.id_,
        content="Some content",
        error=False,
    )

    converted = ChatMessage.from_tool_call_result(tool_call_result)

    assert converted.role == "tool"
    assert converted.content == tool_call_result.model_dump_json(indent=4)
