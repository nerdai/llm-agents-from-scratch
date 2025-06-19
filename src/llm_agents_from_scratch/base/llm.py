"""Base LLM"""

from abc import ABC, abstractmethod

from llm_agents_from_scratch.data_structures import ChatMessage, CompleteResult


class BaseLLM(ABC):
    """Base LLM Class."""

    @abstractmethod
    async def complete(self, prompt: str) -> CompleteResult:
        """Text Complete.

        Args:
            prompt (str): The prompt the LLM should use as input.

        Returns:
            str: The completion of the prompt.
        """

    @abstractmethod
    async def chat(self, chat_messages: list[ChatMessage]) -> ChatMessage:
        """Chat interface.

        Args:
            chat_messages (list[ChatMessage]): chat history.

        Returns:
            ChatMessage: The response of the LLM structured as a `ChatMessage`.
        """
