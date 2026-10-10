"""Errors for the A2A subsystem."""

from .core import LLMAgentsFromScratchError


class A2AError(LLMAgentsFromScratchError):
    """Base error for all A2A-related exceptions."""

    pass


class A2AAgentCardMissingInterfaceError(A2AError):
    """Raised when an AgentCard declares no supported interfaces."""

    pass
