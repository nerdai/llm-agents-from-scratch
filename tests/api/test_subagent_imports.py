import importlib

import pytest

from llm_agents_from_scratch.subagents import __all__ as _subagents_all


@pytest.mark.parametrize("name", _subagents_all)
def test_subagents_all_importable(name: str) -> None:
    """Tests all names listed in subagents __all__ are importable."""
    mod = importlib.import_module("llm_agents_from_scratch.subagents")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None
