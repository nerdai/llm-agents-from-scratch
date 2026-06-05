"""Errors and warnings for episodic memory."""

from .core import LLMAgentsFromScratchWarning


class EpisodeNotFoundWarning(LLMAgentsFromScratchWarning):
    """Issued when delete()/update() targets an episode ID not in the store."""

    pass
