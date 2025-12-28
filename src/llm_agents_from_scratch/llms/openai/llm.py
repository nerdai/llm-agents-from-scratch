"""BONUS Material: OpenAI LLM."""

from typing import TYPE_CHECKING, Any, Sequence

from llm_agents_from_scratch.base.llm import LLM, StructuredOutputType
from llm_agents_from_scratch.base.tool import Tool
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    CompleteResult,
    ToolCallResult,
)
from llm_agents_from_scratch.utils import check_extra_was_installed

from .utils import (
    chat_message_to_openai_response_input_param,
    openai_response_to_chat_message,
    tool_to_openai_tool,
)

if TYPE_CHECKING:
    from openai.types.responses import (
        ParsedResponse,
        Response,
        ResponseInputItemParam,
    )


class OpenAILLM(LLM):
    """OpenAI LLM integration."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Create an OpenAILLM instance.

        Args:
            model (str): The name of the OpenAI LLM.
            api_key (str | None, optional): An OpenAI api key. Defaults to None.
                If None, will fallback to OpenAI's api key resolution, looking
                for an OPENAI_API_KEY env var.
            **kwargs (Any): Additional keyword arguments. Passed to the
                construction of an ~openai.AsyncOpenAI
        """
        check_extra_was_installed(extra="openai", packages="openai")
        from openai import AsyncOpenAI  # noqa: PLC0415

        # Avoid passing duplicate `api_key` if provided both explicitly and
        # in kwargs.
        kwargs.pop("api_key", None)
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, **kwargs)

    async def complete(self, prompt: str, **kwargs: Any) -> CompleteResult:
        """Implements complete LLM interaction mode."""
        response: "Response" = await self.client.responses.create(
            model=self.model,
            input=prompt,
            **kwargs,
        )
        return CompleteResult(response=str(response.output_text), prompt=prompt)

    async def structured_output(
        self,
        prompt: str,
        mdl: type[StructuredOutputType],
        **kwargs: Any,
    ) -> StructuredOutputType:
        """Implements structured output LLM interaction mode.

        Args:
            prompt (str): The prompt to elicit the structured output response.
            mdl (type[StructuredOutputType]): The ~pydantic.BaseModel to output.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            StructuredOutputType: The structured output as the specified `mdl`
                type.
        """
        response: "ParsedResponse" = await self.client.responses.parse(
            model=self.model,
            input=prompt,
            text_format=mdl,
            **kwargs,
        )
        return response.output_parsed  # type: ignore[no-any-return]

    def _prepare_input_and_instructions_from_history(
        self,
        chat_history: Sequence[ChatMessage],
    ) -> tuple[list["ResponseInputItemParam"], str | None]:
        """Prepare response inputs and instructions.

        With OpenAI Responses API system messages can be supplied in
        instructions params. This helper extracts system messages from the chat
        history and prepares instructions by concatenating the content of such
        messages.

        Returns:
            tuple[list[openai.ResponseInputParam], str | None]: The input and
                instructions when invoking openai.client.responses.create()
        """
        context = []
        instruction_messages = []
        for cm in chat_history:
            if cm.role == "system":
                instruction_messages.append(cm.content)
            else:
                context.extend(chat_message_to_openai_response_input_param(cm))
        instructions = (
            "/n".join(instruction_messages) if instruction_messages else None
        )
        return context, instructions

    async def chat(
        self,
        input: str,
        chat_history: Sequence[ChatMessage] | None = None,
        tools: Sequence[Tool] | None = None,
        **kwargs: Any,
    ) -> tuple[ChatMessage, ChatMessage]:
        """Implements chat LLM interaction mode.

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
        # prepare chat history and instructions
        chat_history = chat_history or []
        context, instructions = (
            self._prepare_input_and_instructions_from_history(
                chat_history,
            )
        )

        user_message = ChatMessage(role="user", content=input)
        context.extend(
            chat_message_to_openai_response_input_param(user_message),
        )

        # prepare tools
        openai_tools = (
            [tool_to_openai_tool(t) for t in tools] if tools else None
        )

        response = await self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=context,
            tools=openai_tools,
            **kwargs,
        )
        return user_message, openai_response_to_chat_message(response)

    async def continue_chat_with_tool_results(
        self,
        tool_call_results: Sequence[ToolCallResult],
        chat_history: Sequence[ChatMessage],
        tools: Sequence[Tool] | None = None,
        **kwargs: Any,
    ) -> tuple[list[ChatMessage], ChatMessage]:
        """Implements continue chat with tool results.

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
        # prepare input from chat history
        chat_history = chat_history or []
        context_from_chat_history, instructions = (
            self._prepare_input_and_instructions_from_history(
                chat_history,
            )
        )

        # prepare input from tool call results
        tool_messages = [
            ChatMessage.from_tool_call_result(tc) for tc in tool_call_results
        ]
        context_from_tool_messages = []
        for tm in tool_messages:
            context_from_tool_messages.extend(
                chat_message_to_openai_response_input_param(tm),
            )

        openai_response_input_params = (
            context_from_chat_history + context_from_tool_messages
        )

        # prepare tools
        openai_tools = (
            [tool_to_openai_tool(t) for t in tools] if tools else None
        )

        # send response
        response = await self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=openai_response_input_params,
            tools=openai_tools,
            **kwargs,
        )

        return tool_messages, openai_response_to_chat_message(response)
