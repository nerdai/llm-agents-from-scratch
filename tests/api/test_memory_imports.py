import importlib

import pytest

from llm_agents_from_scratch.memory import __all__ as _memory_all


@pytest.mark.parametrize("name", _memory_all)
def test_memory_all_importable(name: str) -> None:
    """Tests that all names listed in memory __all__ are importable."""
    mod = importlib.import_module("llm_agents_from_scratch.memory")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None
