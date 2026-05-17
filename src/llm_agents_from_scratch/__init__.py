"""Build an LLM agent from scratch."""

from llm_agents_from_scratch._version import VERSION

# Disable the F403 warning for wildcard imports
# ruff: noqa: F403, F401
from .agent import *
from .agent import __all__ as _agent_all
from .memory import *
from .memory import __all__ as _memory_all
from .skills import *
from .skills import __all__ as _skills_all
from .tools import *
from .tools import __all__ as _tool_all

__version__ = VERSION


__all__ = sorted(  # noqa: PLE0605
    _agent_all + _memory_all + _skills_all + _tool_all,
)
