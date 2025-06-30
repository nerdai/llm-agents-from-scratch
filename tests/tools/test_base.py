from llm_agents_from_scratch.base.tool import AsyncBaseTool, BaseTool


def test_base_abstract_attr() -> None:
    """Tests abstract methods in base class."""
    abstract_methods = BaseTool.__abstractmethods__

    assert "name" in abstract_methods
    assert "description" in abstract_methods
    assert "parameters_json_schema" in abstract_methods
    assert "__call__" in abstract_methods


def test_async_base_abstract_attr() -> None:
    """Tests abstract methods in async base class."""
    abstract_methods = AsyncBaseTool.__abstractmethods__

    assert "name" in abstract_methods
    assert "description" in abstract_methods
    assert "parameters_json_schema" in abstract_methods
    assert "__call__" in abstract_methods
