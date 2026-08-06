"""A2A client-side module — dispatching to peer A2A agents."""

from .spec import A2AAgentSpec
from .tools import UseA2AAgentTool

__all__ = [
    "A2AAgentSpec",
    "UseA2AAgentTool",
]
