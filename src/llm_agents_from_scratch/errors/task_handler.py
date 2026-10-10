"""Errors and warnings for TaskHandler."""

from .core import LLMAgentsFromScratchError, LLMAgentsFromScratchWarning


class TaskHandlerError(LLMAgentsFromScratchError):
    """Base error for all TaskHandler-related exceptions."""

    pass


class RecordMemoryError(TaskHandlerError):
    """Raised when record_memory() is called with invalid arguments."""

    pass


class TaskHandlerWarning(LLMAgentsFromScratchWarning):
    """Base warning for all TaskHandler-related warnings."""

    pass


class MemoryRecallWarning(TaskHandlerWarning):
    """Emitted when a memory's recall() fails; the task runs without it."""

    pass


class MemoryRecordWarning(TaskHandlerWarning):
    """Emitted when a memory's record() fails; the task still settles."""

    pass
