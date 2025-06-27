from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ollama import ChatResponse, GenerateResponse
from ollama import Message as OllamaMessage

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ChatRole,
    ToolCall,
)
from llm_agents_from_scratch.llms.ollama import OllamaLLM
from llm_agents_from_scratch.llms.ollama.utils import (
    chat_message_to_ollama_message,
    ollama_message_to_chat_message,
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
                OllamaMessage.ToolCall.Function(
                    name="a fake tool",
                    arguments={"arg1": 1},
                ),
            ],
        ),
    )
    mock_instance.chat = mock_chat
    mock_async_client_class.return_value = mock_instance


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
