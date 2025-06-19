from llm_agents_from_scratch.base.llm import BaseLLM


def test_base_abstract_attr() -> None:
    abstract_methods = BaseLLM.__abstractmethods__

    assert "complete" in abstract_methods
    assert "chat" in abstract_methods
