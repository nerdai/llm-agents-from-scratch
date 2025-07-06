import importlib

import pytest

from llm_agents_from_scratch.tools import __all__ as _tools_all


@pytest.mark.parametrize("name", _tools_all)
def test_tools_all_importable(name: str) -> None:
    """Tests that all names listed in tools __all__ are importable."""
    mod = importlib.import_module("llm_agents_from_scratch.tools")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None
