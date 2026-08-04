"""Subagents module."""

from . import recipes
from .recipes import explore_subagent, general_subagent
from .spec import SubAgentSpec
from .tools import UseSubAgentTool

__all__ = [
    "SubAgentSpec",
    "UseSubAgentTool",
    "explore_subagent",
    "general_subagent",
    "recipes",
]
