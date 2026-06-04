"""Errors for TaskHandler."""

from .core import LLMAgentsFromScratchError


class TaskHandlerError(LLMAgentsFromScratchError):
    """Base error for all TaskHandler-related exceptions."""

    pass


class RecordMemoryError(TaskHandlerError):
    """Raised when record_memory() is called with invalid arguments."""

    pass
