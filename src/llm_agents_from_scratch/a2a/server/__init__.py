"""A2A server-side module — serving an LLMAgent as an A2A peer."""

from .executor import LLMAgentA2AExecutor, build_agent_card
from .streaming_executor import (
    StreamingLLMAgentA2AExecutor,
    build_streaming_agent_card,
)

__all__ = [
    "LLMAgentA2AExecutor",
    "StreamingLLMAgentA2AExecutor",
    "build_agent_card",
    "build_streaming_agent_card",
]
