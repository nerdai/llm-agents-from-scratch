"""BONUS Material: OpenAI LLM."""

from typing import Any

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures.llm import CompleteResult
from llm_agents_from_scratch.utils import check_extra_was_installed


class OpenAILLM(BaseLLM):
    """OpenAI LLM integration."""

    def __init__(
        self,
        model: str,
        /,
        *,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Instantiate an OpenAILLM instance."""
        check_extra_was_installed(extra="openai", packages="openai")
        from openai import AsyncOpenAI  # noqa: PLC0415

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key)

    async def complete(self, prompt: str, **kwargs: Any) -> CompleteResult:
        """Implements complete."""
        response = await self.client.responses.create(
            model=self.model,
            input=prompt,
            **kwargs,
        )
        return CompleteResult(response=response, prompt=prompt)
