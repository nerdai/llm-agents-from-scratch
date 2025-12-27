"""Unit tests for OpenAILLM."""

from importlib.util import find_spec
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ToolCall,
    ToolCallResult,
)
from llm_agents_from_scratch.llms.openai import OpenAILLM
from llm_agents_from_scratch.llms.openai.errors import DataConversionError
from llm_agents_from_scratch.llms.openai.utils import (
    chat_message_to_openai_response_input_param,
    tool_to_openai_tool,
)
from llm_agents_from_scratch.tools import SimpleFunctionTool

openai_installed = bool(find_spec("openai"))

TEST_DATA_PATH = Path(__file__).parents[1] / "_test_data" / "openai"


def test_openai_llm_class() -> None:
    names_of_base_classes = [b.__name__ for b in OpenAILLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.skipif(not openai_installed, reason="openai is not installed")
@patch("openai.AsyncOpenAI")
def test_init(mock_async_client_class: MagicMock) -> None:
    """Tests init of OpenAILLM."""
    mock_instance = MagicMock()
    mock_async_client_class.return_value = mock_instance
    llm = OpenAILLM("gpt-5.2", timeout=3000, max_retries=2)

    assert llm.model == "gpt-5.2"
    assert llm.client == mock_instance
    mock_async_client_class.assert_called_once_with(
        api_key=None,
        timeout=3000,
        max_retries=2,
    )


@pytest.mark.skipif(not openai_installed, reason="openai is not installed")
@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_complete(mock_async_client_class: MagicMock) -> None:
    """Test complete method."""
    from openai.types.responses import Response  # noqa: PLC0415

    # load test data
    with open(TEST_DATA_PATH / "mock_response_for_complete.json", "r") as f:
        mock_response_data = f.read()

    # arrange mocks
    mock_instance = MagicMock()
    mock_generate = AsyncMock()
    mock_generate.return_value = Response.model_validate_json(
        mock_response_data,
    )
    mock_instance.responses.create = mock_generate
    mock_async_client_class.return_value = mock_instance

    llm = OpenAILLM("gpt-5.2")

    # act
    result = await llm.complete("fake prompt")

    # assert
    assert result.response == "fake response"
    assert result.prompt == "fake prompt"


@pytest.mark.skipif(not openai_installed, reason="openai is not installed")
@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_structured_output(mock_async_client_class: MagicMock) -> None:
    """Test structured_output method."""
    from openai.lib._parsing._responses import parse_response  # noqa: PLC0415
    from openai.types.responses import Response  # noqa: PLC0415

    # load test data
    with open(
        TEST_DATA_PATH / "mock_response_for_structured_output.json",
        "r",
    ) as f:
        mock_response_data = f.read()

    # Structured output type
    class Pet(BaseModel):
        animal: str
        name: str

    # arrange mocks
    mock_instance = MagicMock()
    mock_parse = AsyncMock()
    mock_response = Response.model_validate_json(
        mock_response_data,
    )
    mock_parse.return_value = parse_response(
        text_format=Pet,
        input_tools=None,
        response=mock_response,
    )
    mock_instance.responses.parse = mock_parse
    mock_async_client_class.return_value = mock_instance

    llm = OpenAILLM("gpt-5.2")

    # act
    new_pet = await llm.structured_output("Generate a pet.", mdl=Pet)

    assert isinstance(new_pet, Pet)
    assert new_pet.animal == "cat"
    assert new_pet.name == "Whiskers"
    mock_parse.assert_awaited_once_with(
        model="gpt-5.2",
        input="Generate a pet.",
        text_format=Pet,
    )
    mock_async_client_class.assert_called_once()


@pytest.mark.skipif(not openai_installed, reason="openai is not installed")
@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_chat_with_no_tool_results(
    mock_async_client_class: MagicMock,
) -> None:
    """Test chat method."""
    from openai.types.responses import Response  # noqa: PLC0415

    # load test data
    with open(
        TEST_DATA_PATH / "mock_response_for_chat.json",
        "r",
    ) as f:
        mock_response_data = f.read()

    # arrange mocks
    mock_instance = MagicMock()
    mock_create = AsyncMock()
    mock_response = Response.model_validate_json(
        mock_response_data,
    )

    mock_create.return_value = mock_response
    mock_instance.responses.create = mock_create
    mock_async_client_class.return_value = mock_instance

    llm = OpenAILLM("gpt-5.2")

    # act
    user_message, response_message = await llm.chat("Some new input.")

    # assert
    assert user_message.role == "user"
    assert user_message.content == "Some new input."
    assert response_message.role == "assistant"
    assert response_message.content == "Hello! How can I help you today?"
    mock_create.assert_awaited_once_with(
        model="gpt-5.2",
        instructions=None,
        input=[
            {"type": "message", "content": "Some new input.", "role": "user"},
        ],
        tools=None,
    )
    mock_async_client_class.assert_called_once()


@pytest.mark.skipif(not openai_installed, reason="openai is not installed")
@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_chat_with_tool_results(
    mock_async_client_class: MagicMock,
) -> None:
    """Test chat method."""
    from openai.types.responses import Response  # noqa: PLC0415

    # load test data
    with open(
        TEST_DATA_PATH / "mock_response_for_chat_with_tool_calls.json",
        "r",
    ) as f:
        mock_response_data = f.read()

    # arrange mocks
    mock_instance = MagicMock()
    mock_create = AsyncMock()
    mock_response = Response.model_validate_json(
        mock_response_data,
    )

    mock_create.return_value = mock_response
    mock_instance.responses.create = mock_create
    mock_async_client_class.return_value = mock_instance

    llm = OpenAILLM("gpt-5.2")

    def get_weather(
        location: str,
        unit: Literal["celsius", "fahrenheit"],
    ) -> float:
        """Get the current weather for a location"""
        return 42.0

    get_weather_tool = SimpleFunctionTool(get_weather)

    # act
    user_message, response_message = await llm.chat(
        "Some new input.",
        chat_history=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is 42 + 0?"),
            ChatMessage(role="assistant", content="42"),
        ],
        tools=[get_weather_tool],
    )

    # assert
    assert user_message.role == "user"
    assert user_message.content == "Some new input."
    assert response_message.role == "assistant"
    assert response_message.content == ""
    assert response_message.tool_calls[0] == ToolCall(
        id_="call_xyz789",
        tool_name="get_weather",
        arguments={"location": "San Francisco", "unit": "celsius"},
    )
    mock_create.assert_awaited_once_with(
        model="gpt-5.2",
        instructions="You are a helpful assistant.",
        input=[
            {"type": "message", "content": "What is 42 + 0?", "role": "user"},
            {"type": "message", "content": "42", "role": "assistant"},
            {"type": "message", "content": "Some new input.", "role": "user"},
        ],
        tools=[tool_to_openai_tool(get_weather_tool)],
    )
    mock_async_client_class.assert_called_once()


@pytest.mark.skipif(not openai_installed, reason="openai is not installed")
@pytest.mark.asyncio
@patch("openai.AsyncOpenAI")
async def test_continue_chat_with_tool_results(
    mock_async_client_class: MagicMock,
) -> None:
    """Test continue_chat_with_tool_results method."""
    from openai.types.responses import Response  # noqa: PLC0415

    # load test data
    with open(
        TEST_DATA_PATH
        / "mock_response_for_continue_chat_with_tool_results.json",
        "r",
    ) as f:
        mock_response_data = f.read()

    # arrange mocks
    mock_instance = MagicMock()
    mock_create = AsyncMock()
    mock_response = Response.model_validate_json(
        mock_response_data,
    )

    mock_create.return_value = mock_response
    mock_instance.responses.create = mock_create
    mock_async_client_class.return_value = mock_instance

    llm = OpenAILLM("gpt-5.2")

    # act
    tool_call = ToolCall(
        tool_name="a fake tool",
        arguments={"arg1": 1},
    )
    tool_call_results = [
        ToolCallResult(
            tool_call_id=tool_call.id_,
            content="Some content",
            error=False,
        ),
    ]
    (
        tool_messages,
        response_message,
    ) = await llm.continue_chat_with_tool_results(
        tool_call_results=tool_call_results,
        chat_history=[],
    )

    # assert
    assert len(tool_messages) == len(tool_call_results)
    assert response_message.role == "assistant"
    assert response_message.content == "Thank you for the tool results."
    mock_create.assert_awaited_once_with(
        model="gpt-5.2",
        instructions=None,
        input=[
            chat_message_to_openai_response_input_param(
                ChatMessage.from_tool_call_result(tool_result),
            )
            for tool_result in tool_call_results
        ],
        tools=None,
    )
    mock_async_client_class.assert_called_once()


def test_chat_message_to_openai_response_input_param_raises_error() -> None:
    """Tests chat_message_to_openai_response_input_param raises error.

    This helper should raise error if unable to build a ToolCallResult from
    the content of a ChatMessage with role set to TOOL.
    """
    # invalid chat message
    invalid_chat_message = ChatMessage(
        role="tool",
        content="This should be valid ToolCallResult data.",
    )
    msg = (
        "An error occured converting a ChatMessage "
        "to an openai.ResponseInputParam. "
    )

    with pytest.raises(DataConversionError, match=msg):
        chat_message_to_openai_response_input_param(invalid_chat_message)
