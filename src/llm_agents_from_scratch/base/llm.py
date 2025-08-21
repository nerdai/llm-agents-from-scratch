"""Base LLM."""

from abc import ABC, abstractmethod
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel

from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    CompleteResult,
    ToolCallResult,
)

from .tool import Tool

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
        chat_history: Sequence[ChatMessage] | None = None,
        tools: Sequence[Tool] | None = None,
        **kwargs: Any,
    ) -> tuple[ChatMessage, ChatMessage]:
        """Chat interface.

        Args:
            input (str): The user's current input.
            chat_history (Sequence[ChatMessage]|None, optional): chat history.
            tools (Sequence[BaseTool]|None, optional): tools that the LLM
                can call.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            tuple[ChatMessage, ChatMessage]: A tuple of ChatMessage with the
                first message corresponding to the ChatMessage created from the
                supplied input string, and the second ChatMessage is the
                response from the LLM structured.
        """

    @abstractmethod
    async def continue_chat_with_tool_results(
        self,
        tool_call_results: Sequence[ToolCallResult],
        chat_history: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> tuple[list[ChatMessage], ChatMessage]:
        """Continue a chat by submitting tool call results.

        Args:
            tool_call_results (Sequence[ToolCallResult]):
                Tool call results.
            chat_history (Sequence[ChatMessage]): The chat history.
                Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            tuple[list[ChatMessage], ChatMessage]: A tuple whose first element
                is a list of ChatMessage objects corresponding to the
                supplied ToolCallResult converted objects. The second element
                is the response ChatMessage from the LLM.
        """
