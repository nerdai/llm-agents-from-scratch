"""Base LLM."""

from abc import ABC, abstractmethod
from typing import Any, Sequence

from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    CompleteResult,
    ToolCallResult,
)

from .tool import BaseTool


class BaseLLM(ABC):
    """Base LLM Class."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs: Any) -> CompleteResult:
        """Text Complete.

        Args:
            prompt (str): The prompt the LLM should use as input.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            str: The completion of the prompt.
        """

    @abstractmethod
    async def chat(
        self,
        query: str,
        chat_messages: list[ChatMessage] | None = None,
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Chat interface.

        Args:
            query (str): The user's current input.
            chat_messages (list[ChatMessage]|None, optional): chat history.
            tools (list[BaseTool]|None, optional): tools that the LLM can call.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ChatMessage: The response of the LLM structured as a `ChatMessage`.
        """

    @abstractmethod
    async def continue_conversation_with_tool_results(
        self,
        tool_call_results: ToolCallResult | Sequence[ToolCallResult],
        chat_messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatMessage:
        """Continue a conversation submitting tool call results.

        Args:
            tool_call_results (ToolCallResult | Sequence[ToolCallResult]):
                Tool call results. Can be a single one or a sequence of them.
            chat_messages (Sequence[ChatMessage]): The chat history.
                Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ChatMessage: The response of the LLM structured as a `ChatMessage`.
        """
