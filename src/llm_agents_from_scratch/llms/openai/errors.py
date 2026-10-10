"""Errors for OpenAI integration."""

from llm_agents_from_scratch.errors.core import LLMAgentsFromScratchError


class OpenAIIntegrationError(LLMAgentsFromScratchError):
    """Base error for all OpenAI integration exceptions."""

    pass


class DataConversionError(OpenAIIntegrationError):
    """Errors related to converting data structures."""

    pass


class StructuredOutputError(OpenAIIntegrationError):
    """The model returned no parseable structured output."""

    pass


__all__ = [
    "OpenAIIntegrationError",
    "DataConversionError",
    "StructuredOutputError",
]
