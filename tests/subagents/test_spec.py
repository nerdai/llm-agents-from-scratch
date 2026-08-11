"""Unit tests for SubAgentSpec."""

import pytest
from pydantic import ValidationError

from llm_agents_from_scratch.agent import LLMAgentBuilder
from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures.skill import SkillScope
from llm_agents_from_scratch.subagents import SubAgentSpec

MAX_STEPS = 10


def test_subagentspec_init(mock_llm: BaseLLM) -> None:
    """Tests SubAgentSpec initialises with required and optional fields."""
    builder = LLMAgentBuilder(llm=mock_llm)
    spec = SubAgentSpec(
        name="researcher",
        description="Searches the web and summarises findings.",
        builder=builder,
    )

    assert spec.name == "researcher"
    assert spec.description == "Searches the web and summarises findings."
    assert spec.builder is builder
    assert spec.max_steps is None
    assert spec.skills_scopes is None
    assert spec.explicit_only_skills is None


def test_subagentspec_skills_scopes(mock_llm: BaseLLM) -> None:
    """Tests SubAgentSpec stores an explicit skills_scopes value."""
    spec = SubAgentSpec(
        name="coder",
        description="Writes and runs Python code.",
        builder=LLMAgentBuilder(llm=mock_llm),
        skills_scopes=[SkillScope.PROJECT],
    )

    assert spec.skills_scopes == [SkillScope.PROJECT]


def test_subagentspec_explicit_only_skills(mock_llm: BaseLLM) -> None:
    """Tests SubAgentSpec stores an explicit explicit_only_skills value."""
    spec = SubAgentSpec(
        name="coder",
        description="Writes and runs Python code.",
        builder=LLMAgentBuilder(llm=mock_llm),
        explicit_only_skills={"stop-at-one"},
    )

    assert spec.explicit_only_skills == {"stop-at-one"}


def test_subagentspec_max_steps(mock_llm: BaseLLM) -> None:
    """Tests SubAgentSpec stores an explicit max_steps value."""
    spec = SubAgentSpec(
        name="coder",
        description="Writes and runs Python code.",
        builder=LLMAgentBuilder(llm=mock_llm),
        max_steps=MAX_STEPS,
    )

    assert spec.max_steps == MAX_STEPS


def test_subagentspec_requires_name(mock_llm: BaseLLM) -> None:
    """Tests SubAgentSpec raises ValidationError when name is missing."""
    with pytest.raises(ValidationError):
        SubAgentSpec(  # type: ignore[call-arg]
            description="no name provided",
            builder=LLMAgentBuilder(llm=mock_llm),
        )


def test_subagentspec_requires_description(mock_llm: BaseLLM) -> None:
    """Tests SubAgentSpec raises ValidationError when description is missing."""
    with pytest.raises(ValidationError):
        SubAgentSpec(  # type: ignore[call-arg]
            name="coder",
            builder=LLMAgentBuilder(llm=mock_llm),
        )


def test_subagentspec_catalog(mock_llm: BaseLLM) -> None:
    """Tests catalog() returns the XML entry for the spec."""
    spec = SubAgentSpec(
        name="researcher",
        description="Searches the web.",
        builder=LLMAgentBuilder(llm=mock_llm),
    )
    entry = spec.catalog()

    assert "<name>researcher</name>" in entry
    assert "<description>Searches the web.</description>" in entry
