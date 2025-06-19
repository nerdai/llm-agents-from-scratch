import pytest

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures import ChatMessage, CompleteResult


class MockBaseLLM(BaseLLM):
    async def complete(self, prompt: str) -> CompleteResult:
        result = "mock complete"
        return CompleteResult(
            response=result, full_response=f"{prompt} {result}"
        )

    async def chat(self, chat_messages: list[ChatMessage]) -> ChatMessage:
        return ChatMessage(role="assistant", content="mock chat response")


@pytest.fixture()
def mock_llm() -> BaseLLM:
    return MockBaseLLM()
