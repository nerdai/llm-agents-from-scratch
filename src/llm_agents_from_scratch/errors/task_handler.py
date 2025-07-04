"""Errors for TaskHandler."""

from .core import LLMAgentsFromScratchError


class TaskHandlerError(LLMAgentsFromScratchError):
    """Base error for all TaskHandler-related exceptions."""

    pass
