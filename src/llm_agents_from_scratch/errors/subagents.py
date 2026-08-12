"""Errors for the subagents subsystem."""

from .core import LLMAgentsFromScratchError


class SubAgentsError(LLMAgentsFromScratchError):
    """Base error for all subagent-related exceptions."""

    pass


class SubAgentNotFoundError(SubAgentsError):
    """Raised when a named subagent is not in the registry."""

    pass
