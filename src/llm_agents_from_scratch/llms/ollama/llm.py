"""Ollama LLM integration."""

from typing import Any, Sequence

from ollama import AsyncClient

from llm_agents_from_scratch.base.llm import LLM, StructuredOutputType
from llm_agents_from_scratch.base.tool import Tool
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    CompleteResult,
    ToolCallResult,
)

from .utils import (
    chat_message_to_ollama_message,
    ollama_message_to_chat_message,
    tool_to_ollama_tool,
)


class OllamaLLM(LLM):
    """Ollama LLM class.

    Integration to `ollama` library for running open source models locally.
    """

    def __init__(
        self,
        model: str,
        host: str | None = None,
        *,
        think: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create an OllamaLLM instance.

        Args:
            model (str): The name of the LLM model.
            host (str | None): Host of running Ollama service. Defaults to None.
            think (bool): Enable/disable thinking mode. Defaults to False.
            **kwargs (Any): Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model = model
        self._client = AsyncClient(host=host)
        self.think = think

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
            think=self.think,
            **kwargs,
        )
        return mdl.model_validate_json(result.message.content)

    async def chat(
        self,
        input: str,
        chat_history: Sequence[ChatMessage] | None = None,
        tools: Sequence[Tool] | None = None,
        **kwargs: Any,
    ) -> tuple[ChatMessage, ChatMessage]:
        """Chat with an Ollama LLM.

        Args:
            input (str): The user's current input.
            chat_history (list[ChatMessage] | None, optional): The chat
                history.
            tools (list[BaseTool] | None, optional): The tools available to the
                LLM.
            return_history (bool): Whether to return the update chat history.
                Defaults to False.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            tuple[ChatMessage, ChatMessage]: A tuple of ChatMessage with the
                first message corresponding to the ChatMessage created from the
                supplied input string, and the second ChatMessage is the
                response from the LLM.
        """
        # prepare chat history
        o_messages = (
            [chat_message_to_ollama_message(cm) for cm in chat_history]
            if chat_history
            else []
        )

        user_message = ChatMessage(role="user", content=input)
        o_messages.append(
            chat_message_to_ollama_message(
                user_message,
            ),
        )

        # prepare tools
        o_tools = [tool_to_ollama_tool(t) for t in tools] if tools else None

        result = await self._client.chat(
            model=self.model,
            messages=o_messages,
            tools=o_tools,
            think=self.think,
        )

        return user_message, ollama_message_to_chat_message(result.message)

    async def continue_chat_with_tool_results(
        self,
        tool_call_results: Sequence[ToolCallResult],
        chat_history: Sequence[ChatMessage],
        tools: Sequence[Tool] | None = None,
        **kwargs: Any,
    ) -> tuple[list[ChatMessage], ChatMessage]:
        """Implements continue_chat_with_tool_results method.

        Args:
            tool_call_results (Sequence[ToolCallResult]): The tool call results.
            chat_history (Sequence[ChatMessage]): The chat history.
            tools (Sequence[BaseTool]|None, optional): tools that the LLM
                can call.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            tuple[list[ChatMessage], ChatMessage]: A tuple whose first element
                is a list of ChatMessage objects corresponding to the
                supplied ToolCallResult converted objects. The second element
                is the response ChatMessage from the LLM.
        """
        # augment chat messages and convert to Ollama messages
        tool_messages = [
            ChatMessage.from_tool_call_result(tc) for tc in tool_call_results
        ]
        o_messages = [
            chat_message_to_ollama_message(cm) for cm in chat_history
        ] + [chat_message_to_ollama_message(tm) for tm in tool_messages]

        # prepare tools
        o_tools = [tool_to_ollama_tool(t) for t in tools] if tools else None

        # send chat request
        o_result = await self._client.chat(
            model=self.model,
            messages=o_messages,
            tools=o_tools,
            think=self.think,
        )

        return tool_messages, ollama_message_to_chat_message(o_result.message)
