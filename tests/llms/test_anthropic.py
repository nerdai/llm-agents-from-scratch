"""Unit tests for AnthropicLLM."""

from importlib.util import find_spec
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ToolCall,
    ToolCallResult,
)
from llm_agents_from_scratch.llms.anthropic import AnthropicLLM
from llm_agents_from_scratch.llms.anthropic.errors import (
    DataConversionError,
    StructuredOutputError,
)
from llm_agents_from_scratch.llms.anthropic.llm import DEFAULT_MAX_TOKENS
from llm_agents_from_scratch.llms.anthropic.utils import (
    AnthropicChatMessage,
    anthropic_message_to_chat_message,
    chat_message_to_anthropic_message_param,
    chat_message_to_tool_result_block,
    tool_to_anthropic_tool,
)
from llm_agents_from_scratch.tools import SimpleFunctionTool

anthropic_installed = bool(find_spec("anthropic"))

SYSTEM = ChatMessage(role="system", content="You are a helpful assistant.")
OVERRIDE_MAX_TOKENS = 7


def _message(*content: dict[str, Any], stop_reason: str = "end_turn") -> Any:
    """Build a real ~anthropic.types.Message from content block dicts."""
    from anthropic.types import Message  # noqa: PLC0415

    return Message.model_validate(
        {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": list(content),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


@pytest.fixture
def mock_client() -> Any:
    """Patch AsyncAnthropic and yield the mocked client instance."""
    with patch("anthropic.AsyncAnthropic") as mock_class:
        instance = MagicMock()
        instance.messages.create = AsyncMock()
        instance.messages.parse = AsyncMock()
        mock_class.return_value = instance
        yield mock_class, instance


def get_weather(
    location: str,
    unit: Literal["celsius", "fahrenheit"],
) -> float:
    """Get the current weather for a location"""
    return 42.0


get_weather_tool = SimpleFunctionTool(get_weather)


def test_anthropic_llm_class() -> None:
    names_of_base_classes = [b.__name__ for b in AnthropicLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
def test_init(mock_client: Any) -> None:
    """Tests init of AnthropicLLM."""
    mock_class, instance = mock_client

    llm = AnthropicLLM("claude-sonnet-5", timeout=3000, max_retries=2)

    assert llm.model == "claude-sonnet-5"
    assert llm.max_tokens == DEFAULT_MAX_TOKENS
    assert llm.client == instance
    mock_class.assert_called_once_with(
        api_key=None,
        timeout=3000,
        max_retries=2,
    )


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
@pytest.mark.asyncio
async def test_complete(mock_client: Any) -> None:
    """Test complete method."""
    _, instance = mock_client
    instance.messages.create.return_value = _message(
        {"type": "text", "text": "fake response"},
    )

    llm = AnthropicLLM("claude-sonnet-5")
    result = await llm.complete("fake prompt")

    assert result.response == "fake response"
    assert result.prompt == "fake prompt"
    instance.messages.create.assert_awaited_once_with(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "fake prompt"}],
        max_tokens=4096,
    )


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
@pytest.mark.asyncio
async def test_max_tokens_call_kwarg_wins(mock_client: Any) -> None:
    """A per-call max_tokens overrides the constructor default."""
    _, instance = mock_client
    instance.messages.create.return_value = _message(
        {"type": "text", "text": "ok"},
    )

    llm = AnthropicLLM("claude-sonnet-5", max_tokens=100)
    await llm.complete("fake prompt", max_tokens=OVERRIDE_MAX_TOKENS)

    assert (
        instance.messages.create.call_args.kwargs["max_tokens"]
        == OVERRIDE_MAX_TOKENS
    )


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
@pytest.mark.asyncio
async def test_structured_output(mock_client: Any) -> None:
    """Test structured_output method."""
    _, instance = mock_client

    class Pet(BaseModel):
        animal: str
        name: str

    instance.messages.parse.return_value = MagicMock(
        parsed_output=Pet(animal="cat", name="Whiskers"),
    )

    llm = AnthropicLLM("claude-sonnet-5")
    new_pet = await llm.structured_output("Generate a pet.", mdl=Pet)

    assert isinstance(new_pet, Pet)
    assert new_pet.animal == "cat"
    assert new_pet.name == "Whiskers"
    instance.messages.parse.assert_awaited_once_with(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "Generate a pet."}],
        output_format=Pet,
        max_tokens=4096,
    )


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
@pytest.mark.asyncio
async def test_structured_output_raises_when_nothing_parsed(
    mock_client: Any,
) -> None:
    """A refusal or truncation leaves parsed_output None; fail loudly."""
    _, instance = mock_client

    class Pet(BaseModel):
        animal: str

    instance.messages.parse.return_value = MagicMock(
        parsed_output=None,
        stop_reason="max_tokens",
    )

    llm = AnthropicLLM("claude-sonnet-5")

    with pytest.raises(StructuredOutputError, match="max_tokens"):
        await llm.structured_output("Generate a pet.", mdl=Pet)


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
@pytest.mark.asyncio
async def test_chat_with_no_tool_results(mock_client: Any) -> None:
    """Test chat method with a system prompt and no tools."""
    _, instance = mock_client
    instance.messages.create.return_value = _message(
        {"type": "text", "text": "Hello! How can I help you today?"},
    )

    llm = AnthropicLLM("claude-sonnet-5")
    user_message, response_message = await llm.chat(
        "Some new input.",
        chat_history=[SYSTEM],
    )

    assert user_message.role == "user"
    assert user_message.content == "Some new input."
    assert response_message.role == "assistant"
    assert response_message.content == "Hello! How can I help you today?"
    assert response_message.tool_calls == []
    instance.messages.create.assert_awaited_once_with(
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "Some new input."}],
        system="You are a helpful assistant.",
        max_tokens=4096,
    )


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
@pytest.mark.asyncio
async def test_chat_with_tool_results(mock_client: Any) -> None:
    """Test chat method with history and tools; response is a tool call."""
    _, instance = mock_client
    instance.messages.create.return_value = _message(
        {
            "type": "tool_use",
            "id": "toolu_xyz789",
            "name": "get_weather",
            "input": {"location": "San Francisco", "unit": "celsius"},
        },
        stop_reason="tool_use",
    )

    llm = AnthropicLLM("claude-sonnet-5")
    user_message, response_message = await llm.chat(
        "Some new input.",
        chat_history=[
            SYSTEM,
            ChatMessage(role="user", content="What is 42 + 0?"),
            ChatMessage(role="assistant", content="42"),
        ],
        tools=[get_weather_tool],
    )

    assert user_message.content == "Some new input."
    assert response_message.content == ""
    assert response_message.tool_calls[0] == ToolCall(
        id_="toolu_xyz789",
        tool_name="get_weather",
        arguments={"location": "San Francisco", "unit": "celsius"},
    )
    instance.messages.create.assert_awaited_once_with(
        model="claude-sonnet-5",
        messages=[
            {"role": "user", "content": "What is 42 + 0?"},
            {"role": "assistant", "content": "42"},
            {"role": "user", "content": "Some new input."},
        ],
        system="You are a helpful assistant.",
        tools=[tool_to_anthropic_tool(get_weather_tool)],
        max_tokens=4096,
    )


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
def test_tool_to_anthropic_tool() -> None:
    """The tool's JSON schema becomes the Messages API input_schema."""
    anthropic_tool = tool_to_anthropic_tool(get_weather_tool)

    assert anthropic_tool == {
        "name": get_weather_tool.name,
        "description": get_weather_tool.description,
        "input_schema": get_weather_tool.parameters_json_schema,
    }


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
@pytest.mark.asyncio
async def test_continue_chat_with_tool_results(mock_client: Any) -> None:
    """Tool results ride back as one user turn of tool_result blocks."""
    _, instance = mock_client
    instance.messages.create.return_value = _message(
        {"type": "text", "text": "Thank you for the tool results."},
    )

    llm = AnthropicLLM("claude-sonnet-5")

    tool_calls = [
        ToolCall(tool_name="a fake tool", arguments={"arg1": 1}),
        ToolCall(tool_name="another fake tool", arguments={"arg1": 2}),
    ]
    asst_msg = ChatMessage(role="assistant", content="", tool_calls=tool_calls)
    tool_call_results = [
        ToolCallResult(tool_call_id=tc.id_, content="Some content", error=False)
        for tc in tool_calls
    ]

    tool_messages, response_message = await llm.continue_chat_with_tool_results(
        tool_call_results=tool_call_results,
        chat_history=[SYSTEM, asst_msg],
    )

    assert len(tool_messages) == len(tool_call_results)
    assert response_message.content == "Thank you for the tool results."
    instance.messages.create.assert_awaited_once_with(
        model="claude-sonnet-5",
        messages=[
            chat_message_to_anthropic_message_param(asst_msg),
            {
                "role": "user",
                "content": [
                    chat_message_to_tool_result_block(tm)
                    for tm in tool_messages
                ],
            },
        ],
        system="You are a helpful assistant.",
        max_tokens=4096,
    )


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
@pytest.mark.asyncio
async def test_continue_chat_with_tool_results_forwards_tools(
    mock_client: Any,
) -> None:
    """Tools stay available on the follow-up turn, as run_step() passes them."""
    _, instance = mock_client
    instance.messages.create.return_value = _message(
        {"type": "text", "text": "Done."},
    )

    llm = AnthropicLLM("claude-sonnet-5")
    tool_call = ToolCall(tool_name="get_weather", arguments={"location": "SF"})
    asst_msg = ChatMessage(role="assistant", content="", tool_calls=[tool_call])

    await llm.continue_chat_with_tool_results(
        tool_call_results=[
            ToolCallResult(tool_call_id=tool_call.id_, content="42.0"),
        ],
        chat_history=[asst_msg],
        tools=[get_weather_tool],
    )

    kwargs = instance.messages.create.call_args.kwargs
    assert kwargs["tools"] == [tool_to_anthropic_tool(get_weather_tool)]
    assert "system" not in kwargs


def test_chat_message_to_anthropic_message_param_with_tool_calls() -> None:
    """Assistant text plus tool calls becomes text + tool_use blocks."""
    tool_call = ToolCall(id_="toolu_1", tool_name="t", arguments={"a": 1})
    msg = ChatMessage(role="assistant", content="hi", tool_calls=[tool_call])

    assert chat_message_to_anthropic_message_param(msg) == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "hi"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "t",
                "input": {"a": 1},
            },
        ],
    }


def test_chat_message_to_anthropic_message_param_with_tool_result() -> None:
    """A lone tool message becomes a user turn wrapping its tool_result."""
    tool_msg = ChatMessage.from_tool_call_result(
        ToolCallResult(tool_call_id="toolu_1", content="42", error=False),
    )

    assert chat_message_to_anthropic_message_param(tool_msg) == {
        "role": "user",
        "content": [chat_message_to_tool_result_block(tool_msg)],
    }


def test_chat_message_to_tool_result_block_raises_error() -> None:
    """A tool message whose content isn't a ToolCallResult fails loudly."""
    invalid_chat_message = ChatMessage(
        role="tool",
        content="This should be valid ToolCallResult data.",
    )

    with pytest.raises(DataConversionError, match="Unable to build"):
        chat_message_to_tool_result_block(invalid_chat_message)


THINKING_BLOCK = {
    "type": "thinking",
    "thinking": "The user wants the weather; call the tool.",
    "signature": "sig_abc123",
}
TOOL_USE_BLOCK = {
    "type": "tool_use",
    "id": "toolu_1",
    "name": "get_weather",
    "input": {"location": "Toronto"},
}


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
def test_anthropic_message_to_chat_message_keeps_raw_blocks() -> None:
    """Every response block is kept, including kinds ChatMessage can't hold."""
    message = _message(THINKING_BLOCK, TOOL_USE_BLOCK, stop_reason="tool_use")

    chat_message = anthropic_message_to_chat_message(message)

    assert isinstance(chat_message, AnthropicChatMessage)
    assert chat_message.content == ""
    assert len(chat_message.tool_calls) == 1
    assert chat_message.raw_content == list(message.content)
    assert [b.type for b in chat_message.raw_content] == [
        "thinking",
        "tool_use",
    ]


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
def test_anthropic_chat_message_replays_raw_blocks_verbatim() -> None:
    """Replay sends the original blocks, signatures and order intact.

    Rebuilding the turn from text + tool_calls would drop the thinking
    block, and the API rejects a follow-up whose assistant turn is
    missing it.
    """
    message = _message(THINKING_BLOCK, TOOL_USE_BLOCK, stop_reason="tool_use")
    chat_message = anthropic_message_to_chat_message(message)

    param = chat_message_to_anthropic_message_param(chat_message)

    assert param["role"] == "assistant"
    blocks = list(param["content"])
    assert [b["type"] for b in blocks] == ["thinking", "tool_use"]
    assert blocks[0]["signature"] == "sig_abc123"
    assert blocks[1]["id"] == "toolu_1"


@pytest.mark.skipif(not anthropic_installed, reason="anthropic not installed")
@pytest.mark.asyncio
async def test_continue_chat_replays_thinking_before_tool_results(
    mock_client: Any,
) -> None:
    """The follow-up carries the assistant turn exactly as it was received."""
    _, instance = mock_client
    instance.messages.create.side_effect = [
        _message(THINKING_BLOCK, TOOL_USE_BLOCK, stop_reason="tool_use"),
        _message({"type": "text", "text": "It is 21.5C in Toronto."}),
    ]

    llm = AnthropicLLM("claude-sonnet-5")
    user_message, response_message = await llm.chat(
        "Weather in Toronto?",
        chat_history=[SYSTEM],
        tools=[get_weather_tool],
    )
    tool_call = response_message.tool_calls[0]

    await llm.continue_chat_with_tool_results(
        tool_call_results=[
            ToolCallResult(tool_call_id=tool_call.id_, content="21.5"),
        ],
        chat_history=[SYSTEM, user_message, response_message],
        tools=[get_weather_tool],
    )

    replayed = instance.messages.create.call_args.kwargs["messages"]
    assert replayed[1]["role"] == "assistant"
    assert replayed[1]["content"] == [
        b.model_dump(exclude_none=True) for b in response_message.raw_content
    ]
    assert replayed[1]["content"][0]["type"] == "thinking"
    assert replayed[2]["content"][0]["tool_use_id"] == "toolu_1"
