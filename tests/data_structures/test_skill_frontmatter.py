"""Unit tests for SkillFrontmatter."""

import pytest
from pydantic import ValidationError

from llm_agents_from_scratch.data_structures.skill import SkillFrontmatter


def make_skill_frontmatter(**kwargs: object) -> SkillFrontmatter:
    """Create a minimal valid SkillFrontmatter, overridable via kwargs."""
    defaults: dict[str, object] = {
        "name": "my-skill",
        "description": "Does something useful.",
    }
    defaults.update(kwargs)
    return SkillFrontmatter(**defaults)  # type: ignore[arg-type]


def test_skill_frontmatter_minimal() -> None:
    """Tests SkillFrontmatter accepts only required fields."""
    info = make_skill_frontmatter()
    assert info.name == "my-skill"
    assert info.description == "Does something useful."
    assert info.license is None
    assert info.compatibility is None
    assert info.metadata is None
    assert info.allowed_tools is None


def test_skill_frontmatter_allowed_tools_alias() -> None:
    """Tests SkillFrontmatter accepts allowed-tools via hyphenated alias."""
    info = SkillFrontmatter(
        name="my-skill",
        description="A skill.",
        **{"allowed-tools": "Read Grep"},  # type: ignore[arg-type]
    )
    assert info.allowed_tools == "Read Grep"


def test_skill_frontmatter_allowed_tools_by_field_name() -> None:
    """Tests SkillFrontmatter accepts allowed_tools via Python field name."""
    info = make_skill_frontmatter(allowed_tools="Read")
    assert info.allowed_tools == "Read"


def test_skill_frontmatter_empty_name_raises() -> None:
    """Tests SkillFrontmatter raises ValidationError for an empty name."""
    with pytest.raises(ValidationError):
        make_skill_frontmatter(name="")


def test_skill_frontmatter_whitespace_name_raises() -> None:
    """Tests SkillFrontmatter raises ValidationError for whitespace name."""
    with pytest.raises(ValidationError):
        make_skill_frontmatter(name="   ")


def test_skill_frontmatter_empty_description_raises() -> None:
    """Tests SkillFrontmatter raises ValidationError for empty description."""
    with pytest.raises(ValidationError):
        make_skill_frontmatter(description="")


def test_skill_frontmatter_whitespace_description_raises() -> None:
    """Tests SkillFrontmatter raises ValidationError for whitespace desc."""
    with pytest.raises(ValidationError):
        make_skill_frontmatter(description="\t\n")
