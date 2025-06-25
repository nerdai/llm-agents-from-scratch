from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.llms.ollama import OllamaLLM


def test_ollama_llm_class() -> None:
    names_of_base_classes = [b.__name__ for b in OllamaLLM.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes
