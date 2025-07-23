"""Errors for LLMAgent."""

from .core import LLMAgentsFromScratchError


class LLMAgentError(LLMAgentsFromScratchError):
    """Base error for all TaskHandler-related exceptions."""

    pass
