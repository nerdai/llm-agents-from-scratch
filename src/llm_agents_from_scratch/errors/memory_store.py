"""Errors and warnings for episodic memory."""

from .core import LLMAgentsFromScratchError, LLMAgentsFromScratchWarning


class MemoryStoreWarning(LLMAgentsFromScratchWarning):
    """Base warning for all memory-store-related warnings."""

    pass


class MaxResultsExceededWarning(MemoryStoreWarning):
    """Emitted when a store returns more results than ``max_results``."""

    pass


class MemoryStoreError(LLMAgentsFromScratchError):
    """Base error for all memory-store-related exceptions."""

    pass


class EpisodeNotFoundError(MemoryStoreError):
    """Raised when delete()/update() targets an episode ID not in the store."""

    pass
