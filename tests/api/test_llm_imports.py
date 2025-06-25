import importlib

import pytest

from llm_agents_from_scratch.llms import __all__ as _llms_all


@pytest.mark.parametrize("name", _llms_all)
def test_llms_all_importable(name: str) -> None:
    """Tests that all names listed in llms __all__ are importable."""
    mod = importlib.import_module("llm_agents_from_scratch.llms")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None
