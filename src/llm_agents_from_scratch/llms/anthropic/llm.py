"""BONUS Material: Anthropic LLM."""

from typing import TYPE_CHECKING, Any, Sequence

from llm_agents_from_scratch.base.llm import LLM, StructuredOutputType
from llm_agents_from_scratch.base.tool import Tool
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ChatRole,
    CompleteResult,
    ToolCallResult,
)
from llm_agents_from_scratch.utils import check_extra_was_installed

from .errors import StructuredOutputError
from .utils import (
    anthropic_message_to_chat_message,
    chat_message_to_anthropic_message_param,
    chat_message_to_tool_result_block,
    tool_to_anthropic_tool,
)

if TYPE_CHECKING:
    from anthropic.types import Message, MessageParam, ToolResultBlockParam

DEFAULT_MAX_TOKENS = 4096


class AnthropicLLM(LLM):
    """Anthropic LLM integration, built on the Messages API.

    Structured output uses the API's native JSON-schema output
    (``client.messages.parse(..., output_format=mdl)``) rather than
    forcing a single tool whose schema is the model: the SDK sends the
    schema, constrains decoding to it, and hands back a parsed instance
    via ``ParsedMessage.parsed_output``. That property is ``None`` when
    the model produced no parseable text (a refusal, or ``max_tokens``
    hit mid-object), which surfaces as ``StructuredOutputError`` instead
    of a bare ``None``.

    ``max_tokens`` is mandatory on every Messages API call, so it is a
    stored per-instance default (mirroring ``OllamaLLM.think``) that any
    call can override by passing its own ``max_tokens`` kwarg.

    Attributes:
        model (str): The name of the Anthropic model.
        max_tokens (int): Default ``max_tokens`` sent on every request.
        client (AsyncAnthropic): The underlying SDK client.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> None:
        """Create an AnthropicLLM instance.

        Args:
            model (str): The name of the Anthropic model.
            api_key (str | None, optional): An Anthropic api key. Defaults
                to None, falling back to the SDK's own resolution, which
                reads the ANTHROPIC_API_KEY env var.
            max_tokens (int, optional): Default ``max_tokens`` for every
                request. Defaults to 4096.
            **kwargs (Any): Additional keyword arguments. Passed to the
                construction of an ~anthropic.AsyncAnthropic
        """
        check_extra_was_installed(extra="anthropic", packages="anthropic")
        from anthropic import AsyncAnthropic  # noqa: PLC0415

        # Avoid passing duplicate `api_key` if provided both explicitly and
        # in kwargs.
        kwargs.pop("api_key", None)
        self.model = model
        self.max_tokens = max_tokens
        self.client = AsyncAnthropic(api_key=api_key, **kwargs)

    def _with_max_tokens_default(
        self,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Applies the instance's `max_tokens` unless the call set its own."""
        kwargs.setdefault("max_tokens", self.max_tokens)
        return kwargs

    async def complete(self, prompt: str, **kwargs: Any) -> CompleteResult:
        """Implements complete LLM interaction mode."""
        message: "Message" = await self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self._with_max_tokens_default(kwargs),
        )
        return CompleteResult(
            response=anthropic_message_to_chat_message(message).content,
            prompt=prompt,
        )

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

        Raises:
            StructuredOutputError: If the model returned nothing parseable
                into `mdl`.
        """
        message = await self.client.messages.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            output_format=mdl,
            **self._with_max_tokens_default(kwargs),
        )
        parsed = message.parsed_output
        if parsed is None:
            raise StructuredOutputError(
                f"No {mdl.__name__} could be parsed from the response "
                f"(stop_reason={message.stop_reason!r}).",
            )
        return parsed

    def _prepare_messages_and_system_from_history(
        self,
        chat_history: Sequence[ChatMessage],
    ) -> tuple[list["MessageParam"], str | None]:
        """Prepare Messages API turns and the system prompt from history.

        System messages are lifted out into the API's `system` param.
        Consecutive `tool` messages are folded into a single `user` turn
        holding all their `tool_result` blocks, since the API expects every
        result for an assistant's tool calls in the turn that immediately
        follows them.

        Returns:
            tuple[list[MessageParam], str | None]: The turns and the system
                prompt (None when the history carries no system message).
        """
        messages: list["MessageParam"] = []
        system_parts: list[str] = []
        pending_results: list["ToolResultBlockParam"] = []

        def _flush_results() -> None:
            if pending_results:
                messages.append(
                    {"role": "user", "content": list(pending_results)},
                )
                pending_results.clear()

        for cm in chat_history:
            if cm.role == ChatRole.SYSTEM:
                system_parts.append(cm.content)
            elif cm.role == ChatRole.TOOL:
                pending_results.append(chat_message_to_tool_result_block(cm))
            else:
                _flush_results()
                messages.append(chat_message_to_anthropic_message_param(cm))
        _flush_results()

        system = "\n".join(system_parts) if system_parts else None
        return messages, system

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
            **kwargs (Any): Additional keyword arguments.

        Returns:
            tuple[ChatMessage, ChatMessage]: A tuple of ChatMessage with the
                first message corresponding to the ChatMessage created from the
                supplied input string, and the second ChatMessage is the
                response from the LLM.
        """
        messages, system = self._prepare_messages_and_system_from_history(
            chat_history or [],
        )
        user_message = ChatMessage(role="user", content=input)
        messages.append(chat_message_to_anthropic_message_param(user_message))

        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [tool_to_anthropic_tool(t) for t in tools]

        message: "Message" = await self.client.messages.create(
            model=self.model,
            messages=messages,
            **self._with_max_tokens_default(kwargs),
        )
        return user_message, anthropic_message_to_chat_message(message)

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
        tool_messages = [
            ChatMessage.from_tool_call_result(tc) for tc in tool_call_results
        ]
        messages, system = self._prepare_messages_and_system_from_history(
            [*chat_history, *tool_messages],
        )

        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [tool_to_anthropic_tool(t) for t in tools]

        message: "Message" = await self.client.messages.create(
            model=self.model,
            messages=messages,
            **self._with_max_tokens_default(kwargs),
        )
        return tool_messages, anthropic_message_to_chat_message(message)
