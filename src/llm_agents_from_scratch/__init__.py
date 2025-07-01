"""Build an LLM agent from scratch."""

from llm_agents_from_scratch._version import VERSION

# Disable the F403 warning for wildcard imports
# ruff: noqa: F403, F401
from .core import *
from .core import __all__ as _core_all

__version__ = VERSION


__all__ = sorted(_core_all)  # noqa: PLE0605
