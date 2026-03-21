"""Unit tests for validate_skill_dir."""

from pathlib import Path

import pytest

from llm_agents_from_scratch.errors import (
    EmptySkillBodyError,
    InvalidFrontmatterError,
    MissingSkillMdError,
    NameMismatchWarning,
    NameTooLongWarning,
)
from llm_agents_from_scratch.skills.discovery import validate_skill_dir

VALID_SKILL_MD = (
    "---\nname: my-skill\ndescription: A test skill.\n---\nBody content."
)


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    """Create a valid skill directory under tmp_path."""
    d = tmp_path / "my-skill"
    d.mkdir()
    (d / "SKILL.md").write_text(VALID_SKILL_MD)
    return d


def test_valid_dir_returns_info_and_no_warnings(skill_dir: Path) -> None:
    """Tests validate_skill_dir returns info and empty warnings for valid dir."""  # noqa: E501
    info, skill_warnings = validate_skill_dir(skill_dir)

    assert info.name == "my-skill"
    assert info.description == "A test skill."
    assert skill_warnings == []


def test_missing_skill_md_raises(tmp_path: Path) -> None:
    """Tests validate_skill_dir raises MissingSkillMdError if no SKILL.md."""
    d = tmp_path / "my-skill"
    d.mkdir()

    with pytest.raises(MissingSkillMdError):
        validate_skill_dir(d)


def test_no_frontmatter_delimiters_raises(skill_dir: Path) -> None:
    """Tests validate_skill_dir raises InvalidFrontmatterError if no ---."""
    (skill_dir / "SKILL.md").write_text("no frontmatter here")

    with pytest.raises(InvalidFrontmatterError):
        validate_skill_dir(skill_dir)


def test_malformed_yaml_raises(skill_dir: Path) -> None:
    """Tests validate_skill_dir raises InvalidFrontmatterError for bad YAML."""
    (skill_dir / "SKILL.md").write_text(
        "---\nname: [unclosed\n---\nBody content.",
    )

    with pytest.raises(InvalidFrontmatterError):
        validate_skill_dir(skill_dir)


def test_missing_required_fields_raises(skill_dir: Path) -> None:
    """Tests validate_skill_dir raises InvalidFrontmatterError if name missing."""  # noqa: E501
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: A test skill.\n---\nBody content.",
    )

    with pytest.raises(InvalidFrontmatterError):
        validate_skill_dir(skill_dir)


def test_empty_body_raises(skill_dir: Path) -> None:
    """Tests validate_skill_dir raises EmptySkillBodyError for empty body."""
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: A test skill.\n---\n",
    )

    with pytest.raises(EmptySkillBodyError):
        validate_skill_dir(skill_dir)


def test_whitespace_body_raises(skill_dir: Path) -> None:
    """Tests validate_skill_dir raises EmptySkillBodyError for whitespace body."""  # noqa: E501
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: A test skill.\n---\n   \n",
    )

    with pytest.raises(EmptySkillBodyError):
        validate_skill_dir(skill_dir)


def test_name_mismatch_returns_warning(tmp_path: Path) -> None:
    """Tests validate_skill_dir returns NameMismatchWarning on name mismatch."""
    d = tmp_path / "other-name"
    d.mkdir()
    (d / "SKILL.md").write_text(VALID_SKILL_MD)

    _, skill_warnings = validate_skill_dir(d)

    assert len(skill_warnings) == 1
    assert isinstance(skill_warnings[0], NameMismatchWarning)


def test_name_too_long_returns_warning(tmp_path: Path) -> None:
    """Tests validate_skill_dir returns NameTooLongWarning for long names."""
    long_name = "a" * 65
    d = tmp_path / long_name
    d.mkdir()
    (d / "SKILL.md").write_text(
        f"---\nname: {long_name}\ndescription: A test skill.\n---\nBody.",
    )

    _, skill_warnings = validate_skill_dir(d)

    assert len(skill_warnings) == 1
    assert isinstance(skill_warnings[0], NameTooLongWarning)


def test_both_cosmetic_warnings_returned(tmp_path: Path) -> None:
    """Tests validate_skill_dir can return multiple cosmetic warnings."""
    long_name = "a" * 65
    d = tmp_path / "other-name"
    d.mkdir()
    (d / "SKILL.md").write_text(
        f"---\nname: {long_name}\ndescription: A test skill.\n---\nBody.",
    )

    _, skill_warnings = validate_skill_dir(d)

    assert len(skill_warnings) == 2  # noqa: PLR2004
    assert any(isinstance(w, NameMismatchWarning) for w in skill_warnings)
    assert any(isinstance(w, NameTooLongWarning) for w in skill_warnings)
