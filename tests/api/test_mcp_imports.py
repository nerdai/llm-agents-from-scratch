import importlib

import pytest

from llm_agents_from_scratch.tools.model_context_protocol import (
    __all__ as _mcp_all,
)


@pytest.mark.parametrize("name", _mcp_all)
def test_llms_all_importable(name: str) -> None:
    """Tests that all names listed in tools.mcp __all__ are importable."""
    mod = importlib.import_module(
        "llm_agents_from_scratch.tools.model_context_protocol",
    )
    attr = getattr(mod, name)

    assert hasattr(mod, name)
    assert attr is not None
