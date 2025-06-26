from ollama import Message as OllamaMessage

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures import ChatMessage, ChatRole
from llm_agents_from_scratch.llms.ollama import OllamaLLM
from llm_agents_from_scratch.llms.ollama.utils import (
    chat_message_to_ollama_message,
    ollama_message_to_chat_message,
)


def test_ollama_llm_class() -> None:
    names_of_base_classes = [b.__name__ for b in OllamaLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


# test converter methods
def test_chat_message_to_ollama_message() -> None:
    """Tests conversion from ChatMessage to ~ollama.Message."""
    messages = [
        ChatMessage(
            role="user",
            content="1",
        ),
        ChatMessage(
            role="assistant",
            content="2",
        ),
    ]

    ollama_messages = [chat_message_to_ollama_message(m) for m in messages]

    assert ollama_messages[0].content == "1"
    assert ollama_messages[0].role == "user"
    assert ollama_messages[1].content == "2"
    assert ollama_messages[1].role == "assistant"


def test_ollama_message_to_chat_message() -> None:
    """Tests conversion from ~ollama.Message to ChatMessage."""
    messages = [
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

    assert converted[0].role == ChatRole.USER
    assert converted[0].content == "1"
    assert converted[1].role == ChatRole.ASSISTANT
    assert converted[1].content == "2"
    assert converted[2].role == ChatRole.TOOL
    assert converted[2].content == "3"
