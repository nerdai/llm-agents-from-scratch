import importlib

import pytest

from llm_agents_from_scratch import __all__ as _root_all


@pytest.mark.parametrize("name", _root_all)
def test_root_names_all_importable(name: str) -> None:
    """Tests that all names listed in root __all__ are importable."""
    mod = importlib.import_module("llm_agents_from_scratch")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None
