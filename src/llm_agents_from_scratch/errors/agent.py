"""Errors for LLMAgent."""

from .core import LLMAgentsFromScratchError


class LLMAgentError(LLMAgentsFromScratchError):
    """Base error for all TaskHandler-related exceptions."""

    pass


class MaxStepsReachedError(LLMAgentError):
    """Raised if the maximum number of steps reached in a run() method call."""

    pass
