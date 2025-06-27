from typing import Any, Sequence

import pytest

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    CompleteResult,
    ToolCallResult,
)


class MockBaseLLM(BaseLLM):
    async def complete(self, prompt: str) -> CompleteResult:
        result = "mock complete"
        return CompleteResult(
            response=result,
            full_response=f"{prompt} {result}",
        )

    async def chat(
        self,
        chat_messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        return ChatMessage(role="assistant", content="mock chat response")

    async def continue_conversation_with_tool_results(
        self,
        tool_call_results: Sequence[ToolCallResult],
        chat_messages: Sequence[ChatMessage],
        **kwargs: Any,
    ):
        return ChatMessage(
            role="assistant",
            content="mock response to tool call result",
        )


@pytest.fixture()
def mock_llm() -> BaseLLM:
    return MockBaseLLM()
