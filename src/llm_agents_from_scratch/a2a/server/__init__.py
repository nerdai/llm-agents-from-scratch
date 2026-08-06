"""A2A server-side module — serving an LLMAgent as an A2A peer."""

from .card import build_agent_card
from .executor import LLMAgentA2AExecutor

__all__ = [
    "LLMAgentA2AExecutor",
    "build_agent_card",
]
