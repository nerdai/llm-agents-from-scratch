"""Build an LLM agent from scratch."""

from llm_agents_from_scratch._version import VERSION

# Disable the F403 warning for wildcard imports
# ruff: noqa: F403, F401
from .agent import *
from .agent import __all__ as _agent_all
from .tools import *
from .tools import __all__ as _tool_all

__version__ = VERSION


__all__ = sorted(_agent_all + _tool_all)  # noqa: PLE0605
