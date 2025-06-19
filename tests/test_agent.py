from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.core import LLMAgent


def test_init(mock_llm: BaseLLM) -> None:
    """tests init of LLMAgent"""

    agent = LLMAgent(llm=mock_llm)

    assert len(agent.tools) == 0
    assert agent.llm == mock_llm
