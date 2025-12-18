"""Unit tests for OpenAILLM."""

from unittest.mock import MagicMock, patch

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.llms.openai import OpenAILLM


def test_openai_llm_class() -> None:
    names_of_base_classes = [b.__name__ for b in OpenAILLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@patch("openai.AsyncOpenAI")
def test_init(mock_async_client_class: MagicMock) -> None:
    """Tests init of OpenAILLM."""
    mock_instance = MagicMock()
    mock_async_client_class.return_value = mock_instance
    llm = OpenAILLM("gpt-5.2")

    assert llm.model == "gpt-5.2"
    assert llm.client == mock_instance
    mock_async_client_class.assert_called_once()
