"""Prompt templates for LLMAgent (TaskHandler)."""

from typing import TypedDict

DEFAULT_SYSTEM_MESSAGE = """You are a helpful assistant."""


class LLMAgentTemplates(TypedDict):
    """Prompt templates dict for LLMAgent."""

    system_message: str


default_templates = LLMAgentTemplates(
    system_message=DEFAULT_SYSTEM_MESSAGE,
)
