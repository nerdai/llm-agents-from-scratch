"""Ollama LLM integration."""

from typing import Any, Sequence

from ollama import AsyncClient

from llm_agents_from_scratch.base.llm import BaseLLM, StructuredOutputType
from llm_agents_from_scratch.base.tool import AsyncBaseTool, BaseTool
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    CompleteResult,
    ToolCallResult,
)

from .utils import (
    chat_message_to_ollama_message,
    ollama_message_to_chat_message,
    tool_call_result_to_chat_message,
    tool_to_ollama_tool,
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

    async def structured_output(
        self,
        prompt: str,
        mdl: type[StructuredOutputType],
        **kwargs: Any,
    ) -> StructuredOutputType:
        """Structured output interface implementation for Ollama LLM.

        Args:
            prompt (str): The prompt to elicit the structured output response.
            mdl (type[StructuredOutputType]): The ~pydantic.BaseModel to output.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            StructuredOutputType: The structured output as the specified `mdl`
                type.
        """
        o_messages = [
            chat_message_to_ollama_message(
                ChatMessage(role="user", content=prompt),
            ),
        ]
        result = await self._client.chat(
            model=self.model,
            messages=o_messages,
            format=mdl.model_json_schema(),
        )
        return mdl.model_validate_json(result.message.content)

    async def chat(
        self,
        input: str,
        chat_messages: list[ChatMessage] | None = None,
        tools: list[BaseTool | AsyncBaseTool] | None = None,
        return_history: bool = False,
        **kwargs: Any,
    ) -> ChatMessage:
        """Chat with an Ollama LLM.

        Args:
            input (str): The user's current input.
            chat_messages (list[ChatMessage] | None, optional): The chat
                history.
            tools (list[BaseTool] | None, optional): The tools available to the
                LLM.
            return_history (bool): Whether to return the update chat history.
                Defaults to False.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ChatMessage: The chat message from the LLM.
        """
        # prepare chat history
        o_messages = (
            [chat_message_to_ollama_message(cm) for cm in chat_messages]
            if chat_messages
            else []
        )

        o_messages.append(
            chat_message_to_ollama_message(
                ChatMessage(role="user", content=input),
            ),
        )

        # prepare tools
        o_tools = [tool_to_ollama_tool(t) for t in tools] if tools else None

        result = await self._client.chat(
            model=self.model,
            messages=o_messages,
            tools=o_tools,
        )

        return ollama_message_to_chat_message(result.message)

    async def continue_conversation_with_tool_results(
        self,
        tool_call_results: Sequence[ToolCallResult],
        chat_messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> list[ChatMessage]:
        """Implements continue_conversation_with_tool_results method.

        Args:
            tool_call_results (Sequence[ToolCallResult]): The tool call results.
            chat_messages (Sequence[ChatMessage]): The chat history.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            list[ChatMessage]: The chat messages that continue the provided
                conversation history. This should include the tool call
                results as chat messages as well as the LLM's response to the
                tool call results.
        """
        # augment chat messages and convert to Ollama messages
        tool_messages = [
            tool_call_result_to_chat_message(tc) for tc in tool_call_results
        ]
        o_messages = [
            chat_message_to_ollama_message(cm) for cm in chat_messages
        ] + [chat_message_to_ollama_message(tm) for tm in tool_messages]

        # send chat request
        o_result = await self._client.chat(
            model=self.model,
            messages=o_messages,
        )

        return tool_messages + [
            ollama_message_to_chat_message(o_result.message),
        ]
