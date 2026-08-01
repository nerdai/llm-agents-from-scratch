"""Unit tests for SubAgentSpec."""

import pytest
from pydantic import ValidationError

from llm_agents_from_scratch.agent import LLMAgent
from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.subagents import SubAgentSpec

MAX_STEPS = 10


def test_subagentspec_init(mock_llm: BaseLLM) -> None:
    """Tests SubAgentSpec initialises with required and optional fields."""
    agent = LLMAgent(llm=mock_llm)
    spec = SubAgentSpec(
        name="researcher",
        description="Searches the web and summarises findings.",
        agent=agent,
    )

    assert spec.name == "researcher"
    assert spec.description == "Searches the web and summarises findings."
    assert spec.agent is agent
    assert spec.max_steps is None


def test_subagentspec_max_steps(mock_llm: BaseLLM) -> None:
    """Tests SubAgentSpec stores an explicit max_steps value."""
    agent = LLMAgent(llm=mock_llm)
    spec = SubAgentSpec(
        name="coder",
        description="Writes and runs Python code.",
        agent=agent,
        max_steps=MAX_STEPS,
    )

    assert spec.max_steps == MAX_STEPS


def test_subagentspec_requires_name(mock_llm: BaseLLM) -> None:
    """Tests SubAgentSpec raises ValidationError when name is missing."""
    with pytest.raises(ValidationError):
        SubAgentSpec(  # type: ignore[call-arg]
            description="no name provided",
            agent=LLMAgent(llm=mock_llm),
        )


def test_subagentspec_requires_description(mock_llm: BaseLLM) -> None:
    """Tests SubAgentSpec raises ValidationError when description is missing."""
    with pytest.raises(ValidationError):
        SubAgentSpec(  # type: ignore[call-arg]
            name="coder",
            agent=LLMAgent(llm=mock_llm),
        )


def test_subagentspec_catalog(mock_llm: BaseLLM) -> None:
    """Tests catalog() returns the XML entry for the spec."""
    spec = SubAgentSpec(
        name="researcher",
        description="Searches the web.",
        agent=LLMAgent(llm=mock_llm),
    )
    entry = spec.catalog()

    assert "<name>researcher</name>" in entry
    assert "<description>Searches the web.</description>" in entry
