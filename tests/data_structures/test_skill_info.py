"""Unit tests for SkillInfo."""

import pytest
from pydantic import ValidationError

from llm_agents_from_scratch.data_structures.skill import SkillInfo


def make_skill_info(**kwargs: object) -> SkillInfo:
    """Create a minimal valid SkillInfo, overridable via kwargs."""
    defaults: dict[str, object] = {
        "name": "my-skill",
        "description": "Does something useful.",
    }
    defaults.update(kwargs)
    return SkillInfo(**defaults)  # type: ignore[arg-type]


def test_skill_info_minimal() -> None:
    """Tests SkillInfo accepts only required fields."""
    info = make_skill_info()
    assert info.name == "my-skill"
    assert info.description == "Does something useful."
    assert info.license is None
    assert info.compatibility is None
    assert info.metadata is None
    assert info.allowed_tools is None


def test_skill_info_allowed_tools_alias() -> None:
    """Tests SkillInfo accepts allowed-tools via hyphenated alias."""
    info = SkillInfo(
        name="my-skill",
        description="A skill.",
        **{"allowed-tools": "Read Grep"},  # type: ignore[arg-type]
    )
    assert info.allowed_tools == "Read Grep"


def test_skill_info_allowed_tools_by_field_name() -> None:
    """Tests SkillInfo accepts allowed_tools via Python field name."""
    info = make_skill_info(allowed_tools="Read")
    assert info.allowed_tools == "Read"


def test_skill_info_empty_name_raises() -> None:
    """Tests SkillInfo raises ValidationError for an empty name."""
    with pytest.raises(ValidationError):
        make_skill_info(name="")


def test_skill_info_whitespace_name_raises() -> None:
    """Tests SkillInfo raises ValidationError for a whitespace-only name."""
    with pytest.raises(ValidationError):
        make_skill_info(name="   ")


def test_skill_info_empty_description_raises() -> None:
    """Tests SkillInfo raises ValidationError for an empty description."""
    with pytest.raises(ValidationError):
        make_skill_info(description="")


def test_skill_info_whitespace_description_raises() -> None:
    """Tests SkillInfo raises ValidationError for whitespace-only desc."""
    with pytest.raises(ValidationError):
        make_skill_info(description="\t\n")
