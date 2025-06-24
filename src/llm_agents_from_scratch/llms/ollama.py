"""Ollama LLM integration."""

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import ChatMessage, CompleteResult


class OllamaLLM(BaseLLM):
    """Ollama LLM class.

    Integration to `ollama` library for running open source models locally.
    """

    async def complete(self, prompt: str) -> CompleteResult:
        raise NotImplementedError

    async def chat(
        self, chat_messages: list[ChatMessage], tools: list[BaseTool]
    ) -> ChatMessage:
        raise NotImplementedError
