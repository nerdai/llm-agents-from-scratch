"""Unit tests for OpenAILLM."""

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.llms.openai import OpenAILLM


def test_openai_llm_class() -> None:
    names_of_base_classes = [b.__name__ for b in OpenAILLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
