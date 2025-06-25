"""Ollama LLM integration."""

from typing import Any

from ollama import AsyncClient

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import ChatMessage, CompleteResult


class OllamaLLM(BaseLLM):
    """Ollama LLM class.

    Integration to `ollama` library for running open source models locally.
    """

    def __init__(self, model: str, *args: Any, **kwargs: Any) -> None:
        """Create an OllamaLLM instance.

        Args:
            model (str): The name of the LLM model.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self._client = AsyncClient()

    async def complete(self, prompt: str, **kwargs: Any) -> CompleteResult:
        """Complete a prompt with an Ollama LLM.

        Args:
            prompt (str): The prompt to complete.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            CompleteResult: The text completion result.
        """
        response = await self._client.generate(model=self.model, prompt=prompt)
        return CompleteResult(
            response=response.response,
            prompt=prompt,
        )

    async def chat(
        self,
        chat_messages: list[ChatMessage],
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Chat with an Ollama LLM.

        Args:
            chat_messages (list[ChatMessage]): The chat history.
            tools (list[BaseTool]): The tools available to the LLM.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ChatMessage: The chat message from the LLM.
        """
        raise NotImplementedError  # pragma: no cover
