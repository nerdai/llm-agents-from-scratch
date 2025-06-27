"""Ollama LLM integration."""

from typing import Any, Sequence

from ollama import AsyncClient

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    CompleteResult,
    ToolCallResult,
)

from .utils import (
    chat_message_to_ollama_message,
    ollama_message_to_chat_message,
    tool_call_result_to_ollama_message,
)


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
        response = await self._client.generate(
            model=self.model,
            prompt=prompt,
            **kwargs,
        )
        return CompleteResult(
            response=response.response,
            prompt=prompt,
        )

    async def chat(
        self,
        query: str,
        chat_messages: list[ChatMessage] | None = None,
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Chat with an Ollama LLM.

        Args:
            query (str): The user's current input.
            chat_messages (list[ChatMessage] | None, optional): The chat
                history.
            tools (list[BaseTool] | None, optional): The tools available to the
                LLM.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ChatMessage: The chat message from the LLM.
        """
        o_messages = [ChatMessage(role="user", content=query)]
        o_messages.extend(
            [chat_message_to_ollama_message(cm) for cm in chat_messages]
            if chat_messages
            else [],
        )

        result = await self._client.chat(model=self.model, messages=o_messages)

        return ollama_message_to_chat_message(result.message)

    async def continue_conversation_with_tool_results(
        self,
        tool_call_results: Sequence[ToolCallResult],
        chat_messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatMessage:
        """Implements continue_conversation_with_tool_results method.

        Args:
            tool_call_results (Sequence[ToolCallResult]): The tool call results.
            chat_messages (Sequence[ChatMessage]): The chat history.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ChatMessage: The chat message from the LLM.
        """
        o_messages = [
            tool_call_result_to_ollama_message(tc) for tc in tool_call_results
        ] + [chat_message_to_ollama_message(cm) for cm in chat_messages]

        result = await self._client.chat(model=self.model, messages=o_messages)

        return ollama_message_to_chat_message(result.message)
