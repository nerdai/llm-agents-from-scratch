from llm_agents_from_scratch.base.memory_store import BaseMemoryStore


def test_base_memory_store_abstract_methods() -> None:
    """Tests abstract methods of BaseMemoryStore."""
    abstract_methods = BaseMemoryStore.__abstractmethods__

    assert "write" in abstract_methods
    assert "count" in abstract_methods
    assert "read_recent" in abstract_methods
    assert "search" in abstract_methods
    assert "summary" in abstract_methods
