"""Unit tests for OpenAILLM."""

from importlib.util import find_spec
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.llms.openai import OpenAILLM

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
    llm = OpenAILLM("gpt-5.2")

    assert llm.model == "gpt-5.2"
    assert llm.client == mock_instance
    mock_async_client_class.assert_called_once()


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
