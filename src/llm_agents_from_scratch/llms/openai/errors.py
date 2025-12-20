"""Errors for OpenAI integration."""

from llm_agents_from_scratch.errors.core import LLMAgentsFromScratchError


class OpenAIIntegrationError(LLMAgentsFromScratchError):
    """Base error for all TaskHandler-related exceptions."""

    pass


class DataConversionError(OpenAIIntegrationError):
    """Errors related to converting data structures."""

    pass


__all__ = ["OpenAIIntegrationError", "DataConversionError"]
