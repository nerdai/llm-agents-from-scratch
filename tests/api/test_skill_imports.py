import importlib

import pytest

from llm_agents_from_scratch.skills import __all__ as _skills_all


@pytest.mark.parametrize("name", _skills_all)
def test_skills_all_importable(name: str) -> None:
    """Tests all names listed in skills __all__ are importable."""
    mod = importlib.import_module("llm_agents_from_scratch.skills")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None
