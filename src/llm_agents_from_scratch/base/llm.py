"""Base LLM."""

from abc import ABC, abstractmethod
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel

from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    CompleteResult,
    ToolCallResult,
)

from .tool import AsyncBaseTool, BaseTool

StructuredOutputType = TypeVar("StructuredOutputType", bound=BaseModel)


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
    async def structured_output(
        self,
        prompt: str,
        mdl: type[StructuredOutputType],
        **kwargs: Any,
    ) -> StructuredOutputType:
        """Structured output interface for returning ~pydantic.BaseModels.

        Args:
            prompt (str): The prompt to elicit the structured output response.
            mdl (type[StructuredOutputType]): The ~pydantic.BaseModel to output.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            StructuredOutputType: The structured output as the specified `mdl`
                type.
        """

    @abstractmethod
    async def chat(
        self,
        input: str,
        chat_messages: Sequence[ChatMessage] | None = None,
        tools: Sequence[BaseTool | AsyncBaseTool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Chat interface.

        Args:
            input (str): The user's current input.
            chat_messages (Sequence[ChatMessage]|None, optional): chat history.
            tools (Sequence[BaseTool]|None, optional): tools that the LLM
                can call.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ChatMessage: The response of the LLM structured as a `ChatMessage`.
        """

    @abstractmethod
    async def continue_conversation_with_tool_results(
        self,
        tool_call_results: Sequence[ToolCallResult],
        chat_messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """Continue a conversation submitting tool call results.

        Args:
            tool_call_results (Sequence[ToolCallResult]):
                Tool call results.
            chat_messages (Sequence[ChatMessage]): The chat history.
                Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            list[ChatMessage]: The chat messages that continue the provided
                conversation history. This should include the tool call
                results as chat messages as well as the LLM's response to the
                tool call results.
        """
