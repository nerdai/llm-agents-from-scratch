"""Errors for the A2A subsystem."""

from .core import LLMAgentsFromScratchError


class A2AError(LLMAgentsFromScratchError):
    """Base error for all A2A-related exceptions."""

    pass


class A2AAgentNotFoundError(A2AError):
    """Raised when a named A2A agent is not in the registry."""

    pass
