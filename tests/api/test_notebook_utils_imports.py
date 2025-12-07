"""Unit tests for importing notebook utils."""

import importlib

import pytest

from llm_agents_from_scratch import notebook_utils


@pytest.mark.parametrize("name", notebook_utils.__all__)
def test_notebook_utils_all_importable(name: str) -> None:
    """Tests that all names listed in notebook_utils __all__ are importable."""
    mod = importlib.import_module("llm_agents_from_scratch.notebook_utils")
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None
