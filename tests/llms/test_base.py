from llm_agents_from_scratch.base.llm import BaseLLM


def test_base_abstract_attr() -> None:
    """Tests abstract methods in base class."""
    abstract_methods = BaseLLM.__abstractmethods__

    assert "complete" in abstract_methods
    assert "chat" in abstract_methods
    assert "structured_output" in abstract_methods
    assert "continue_conversation_with_tool_results" in abstract_methods
