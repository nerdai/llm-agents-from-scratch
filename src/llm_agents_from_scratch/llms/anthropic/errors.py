"""Errors for Anthropic integration."""

from llm_agents_from_scratch.errors.core import LLMAgentsFromScratchError


class AnthropicIntegrationError(LLMAgentsFromScratchError):
    """Base error for all Anthropic integration exceptions."""

    pass


class DataConversionError(AnthropicIntegrationError):
    """Errors related to converting data structures."""

    pass


class StructuredOutputError(AnthropicIntegrationError):
    """The model returned no parseable structured output."""

    pass


__all__ = [
    "AnthropicIntegrationError",
    "DataConversionError",
    "StructuredOutputError",
]
