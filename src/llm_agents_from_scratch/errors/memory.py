"""Errors and warnings for episodic memory."""

from .core import LLMAgentsFromScratchError, LLMAgentsFromScratchWarning


class MemoryStoreWarning(LLMAgentsFromScratchWarning):
    """Base warning for all memory-store-related warnings."""

    pass


class MemoryStoreError(LLMAgentsFromScratchError):
    """Base error for all memory-store-related exceptions."""

    pass


class EpisodeNotFoundWarning(MemoryStoreWarning):
    """Issued when delete()/update() targets an episode ID not in the store."""

    pass
