"""Ollama LLM integration."""

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import ChatMessage, CompleteResult


class OllamaLLM(BaseLLM):
    """Ollama LLM class.

    Integration to `ollama` library for running open source models locally.
    """

    async def complete(self, prompt: str) -> CompleteResult:
        """Complete a prompt with an Ollama LLM.

        Args:
            prompt (str): The prompt to complete.

        Returns:
            CompleteResult: The text completion result.
        """
        raise NotImplementedError

    async def chat(
        self,
        chat_messages: list[ChatMessage],
        tools: list[BaseTool],
    ) -> ChatMessage:
        """Chat with an Ollama LLM.

        Args:
            chat_messages (list[ChatMessage]): The chat history.
            tools (list[BaseTool]): The tools available to the LLM.

        Returns:
            ChatMessage: The chat message from the LLM.
        """
        raise NotImplementedError
