"""Errors for notebook utils."""

from .core import LLMAgentsFromScratchError


class NotebookUtilsError(LLMAgentsFromScratchError):
    """Base error for all notebook-utils-related exceptions."""

    pass


class UnsupportedProviderError(NotebookUtilsError):
    """Raises when an LLM provider is unknown or not yet implemented."""

    pass


__all__ = ["NotebookUtilsError", "UnsupportedProviderError"]
