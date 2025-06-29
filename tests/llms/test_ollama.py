from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ollama import ChatResponse, GenerateResponse
from ollama import Message as OllamaMessage
from ollama import Tool as OllamaTool

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ChatRole,
    ToolCall,
    ToolCallResult,
)
from llm_agents_from_scratch.llms.ollama import OllamaLLM
from llm_agents_from_scratch.llms.ollama.utils import (
    DEFAULT_TOOL_RESPONSE_TEMPLATE,
    chat_message_to_ollama_message,
    get_tool_json_schema,
    ollama_message_to_chat_message,
    tool_call_result_to_ollama_message,
    tool_to_ollama_tool,
)


def test_ollama_llm_class() -> None:
    names_of_base_classes = [b.__name__ for b in OllamaLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@patch("llm_agents_from_scratch.llms.ollama.llm.AsyncClient")
def test_init(mock_async_client_class: MagicMock) -> None:
    """Tests init of OllamaLLM."""
    mock_instance = MagicMock()
    mock_async_client_class.return_value = mock_instance
    llm = OllamaLLM(model="llama3.2")

    assert llm.model == "llama3.2"
    assert llm._client == mock_instance
    mock_async_client_class.assert_called_once()


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.llms.ollama.llm.AsyncClient")
async def test_complete(mock_async_client_class: MagicMock) -> None:
    """Test complete method."""
    # arrange mocks
    mock_instance = MagicMock()
    mock_generate = AsyncMock()
    mock_generate.return_value = GenerateResponse(
        model="llama3.2",
        response="fake response",
    )
    mock_instance.generate = mock_generate
    mock_async_client_class.return_value = mock_instance

    llm = OllamaLLM(model="llama3.2")

    # act
    result = await llm.complete("fake prompt")

    # assert
    assert result.response == "fake response"
    assert result.prompt == "fake prompt"


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.llms.ollama.llm.AsyncClient")
async def test_chat(mock_async_client_class: MagicMock) -> None:
    """Test chat method."""
    # arrange mocks
    mock_instance = MagicMock()
    mock_chat = AsyncMock()
    mock_chat.return_value = ChatResponse(
        model="llama3.2",
        message=OllamaMessage(
            role="assistant",
            content="some fake content",
            tool_calls=[
                OllamaMessage.ToolCall(
                    function=OllamaMessage.ToolCall.Function(
                        name="a_fake_tool",
                        arguments={"arg1": 1},
                    ),
                ),
            ],
        ),
    )
    mock_instance.chat = mock_chat
    mock_async_client_class.return_value = mock_instance

    llm = OllamaLLM(model="llama3.2")

    # act
    result = await llm.chat("Some new input.")

    assert result.role == "assistant"
    assert result.content == "some fake content"
    mock_chat.assert_awaited_once_with(
        model="llama3.2",
        messages=[OllamaMessage(role="user", content="Some new input.")],
        tools=None,
    )
    mock_async_client_class.assert_called_once()


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.llms.ollama.llm.AsyncClient")
async def test_continue_conversation_with_tool_results(
    mock_async_client_class: MagicMock,
) -> None:
    """Test continue_conversation_with_tool_results method."""

    # arrange mocks
    mock_instance = MagicMock()
    mock_chat = AsyncMock()
    mock_chat.return_value = ChatResponse(
        model="llama3.2",
        message=OllamaMessage(
            role="assistant",
            content="Thank you for the tool call results.",
        ),
    )
    mock_instance.chat = mock_chat
    mock_async_client_class.return_value = mock_instance

    llm = OllamaLLM(model="llama3.2")

    # act
    tool_call_results = [
        ToolCallResult(
            tool_call=ToolCall(
                tool_name="a fake tool",
                arguments={"arg1": 1},
            ),
            content="Some content",
            error=False,
        ),
    ]
    result = await llm.continue_conversation_with_tool_results(
        tool_call_results=tool_call_results,
        chat_messages=[],
    )

    assert result.role == "assistant"
    assert result.content == "Thank you for the tool call results."
    mock_chat.assert_awaited_once_with(
        model="llama3.2",
        messages=[tool_call_result_to_ollama_message(tool_call_results[0])],
    )
    mock_async_client_class.assert_called_once()


# test converter methods
def test_chat_message_to_ollama_message() -> None:
    """Tests conversion from ChatMessage to ~ollama.Message."""
    messages = [
        ChatMessage(
            role="system",
            content="0",
        ),
        ChatMessage(
            role="user",
            content="1",
        ),
        ChatMessage(
            role="assistant",
            content="2",
            tool_calls=[
                ToolCall(
                    tool_name="a tool",
                    arguments={
                        "arg1": "1",
                        "arg2": 2,
                    },
                ),
            ],
        ),
        ChatMessage(
            role="tool",
            content="3",
        ),
    ]

    ollama_messages = [chat_message_to_ollama_message(m) for m in messages]

    assert ollama_messages[0].content == "0"
    assert ollama_messages[0].role == "system"
    assert ollama_messages[0].tool_calls is None

    assert ollama_messages[1].content == "1"
    assert ollama_messages[1].role == "user"
    assert ollama_messages[1].tool_calls is None

    assert ollama_messages[2].content == "2"
    assert ollama_messages[2].role == "assistant"
    assert ollama_messages[2].tool_calls[0].function.name == "a tool"
    assert ollama_messages[2].tool_calls[0].function.arguments == {
        "arg1": "1",
        "arg2": 2,
    }

    assert ollama_messages[3].content == "3"
    assert ollama_messages[3].role == "tool"
    assert ollama_messages[3].tool_calls is None


def test_ollama_message_to_chat_message() -> None:
    """Tests conversion from ~ollama.Message to ChatMessage."""
    messages = [
        OllamaMessage(
            role="system",
            content="0",
        ),
        OllamaMessage(
            role="user",
            content="1",
        ),
        OllamaMessage(
            role="assistant",
            content="2",
            tool_calls=[
                OllamaMessage.ToolCall(
                    function=OllamaMessage.ToolCall.Function(
                        name="fake tool",
                        arguments={
                            "fake_param": "1",
                            "another_fake_param": "2",
                        },
                    ),
                ),
            ],
        ),
        OllamaMessage(
            role="tool",
            content="3",
        ),
    ]

    converted = [ollama_message_to_chat_message(m) for m in messages]

    assert converted[0].role == ChatRole.SYSTEM
    assert converted[0].content == "0"
    assert converted[0].tool_calls is None

    assert converted[1].role == ChatRole.USER
    assert converted[1].content == "1"
    assert converted[1].tool_calls is None

    assert converted[2].role == ChatRole.ASSISTANT
    assert converted[2].content == "2"
    assert converted[2].tool_calls[0].tool_name == "fake tool"
    assert converted[2].tool_calls[0].arguments == {
        "fake_param": "1",
        "another_fake_param": "2",
    }

    assert converted[3].role == ChatRole.TOOL
    assert converted[3].content == "3"
    assert converted[3].tool_calls is None


def test_ollama_message_to_chat_message_raises_error() -> None:
    """Test conversion to chat message raises error with invalid role."""
    with pytest.raises(RuntimeError):
        ollama_message_to_chat_message(
            OllamaMessage(
                role="invalid role",
                content="0",
            ),
        )


def test_tool_call_result_to_ollama_message() -> None:
    """Test conversion of tool call result to an ~ollama.Message."""
    tool_call_result = ToolCallResult(
        tool_call=ToolCall(
            tool_name="a fake tool",
            arguments={"arg1": 1},
        ),
        content="Some content",
        error=False,
    )

    converted = tool_call_result_to_ollama_message(tool_call_result)

    assert converted.role == "tool"
    assert converted.content == DEFAULT_TOOL_RESPONSE_TEMPLATE.format(
        tool_name="a fake tool",
        arguments={"arg1": 1},
        tool_call_result="Some content",
    )


class MyTool(BaseTool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "mock description"

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "param 1",
                },
                "param2": {
                    "type": "number",
                    "description": "param 2",
                },
            },
            "required": ["param1"],
        }

    def __call__(
        self,
        tool_call: ToolCall,
        *args,
        **kwargs,
    ) -> ToolCallResult:
        return ToolCallResult(
            tool_call=tool_call,
            content="fake content",
        )


def test_get_tool_json_schema() -> None:
    """Tests util for getting JSON schema of a tool."""
    # arrange
    my_tool = MyTool()

    # act
    schema = get_tool_json_schema(my_tool)

    # assert
    assert schema["type"] == "function"
    assert schema["function"]["name"] == my_tool.name
    assert schema["function"]["description"] == my_tool.description
    assert schema["function"]["parameters"] == my_tool.parameters_json_schema


def test_tool_to_ollama_tool() -> None:
    """Tests tool conversion util method."""
    # arrange
    my_tool = MyTool()

    # act
    converted_tool = tool_to_ollama_tool(my_tool)

    # arrange
    assert isinstance(converted_tool, OllamaTool)
    assert converted_tool.function.name == my_tool.name
    assert converted_tool.function.description == my_tool.description
    assert len(converted_tool.function.parameters.properties) == len(
        my_tool.parameters_json_schema["properties"],
    )
