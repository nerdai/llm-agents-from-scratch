"""BONUS Material: OpenAI LLM."""

from typing import Any

from llm_agents_from_scratch.base.llm import LLM, StructuredOutputType
from llm_agents_from_scratch.data_structures.llm import CompleteResult
from llm_agents_from_scratch.utils import check_extra_was_installed


class OpenAILLM(LLM):
    """OpenAI LLM integration."""

    def __init__(
        self,
        model: str,
        /,
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

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def complete(self, prompt: str, **kwargs: Any) -> CompleteResult:
        """Implements complete LLM interaction mode."""
        response = await self.client.responses.create(
            model=self.model,
            input=prompt,
            **kwargs,
        )
        return CompleteResult(response=response, prompt=prompt)

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
        response = await self.client.responses.parse(
            model=self.model,
            input=prompt,
            text_format=mdl,
            **kwargs,
        )
        return response.output_parsed  # type: ignore[no-any-return]
