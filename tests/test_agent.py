from unittest.mock import MagicMock

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.core import LLMAgent


def test_init(mock_llm: BaseLLM) -> None:
    """tests init of LLMAgent"""

    agent = LLMAgent(llm=mock_llm)

    assert len(agent.tools) == 0
    assert agent.llm == mock_llm


def test_add_tool(mock_llm: BaseLLM) -> None:
    """tests add tool"""

    # arrange
    tool = MagicMock()
    agent = LLMAgent(llm=mock_llm)

    # act
    agent.add_tool(tool)

    # assert
    assert agent.tools == [tool]
