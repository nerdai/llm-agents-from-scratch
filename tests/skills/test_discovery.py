"""Unit tests for skill discovery."""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_agents_from_scratch.errors import (
    SkillSkippedWarning,
    SkillValidationError,
    SkillValidationWarning,
)
from llm_agents_from_scratch.skills.constants import SKILLS_PATHS
from llm_agents_from_scratch.skills.discovery import (
    discover_skills,
)
from llm_agents_from_scratch.skills.skill import Skill


@pytest.fixture
def skills_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a skill under tmp_path and point SKILLS_PATHS at it.

    Uses monkeypatch.setitem to temporarily replace the project scope paths
    in SKILLS_PATHS with tmp_path, so no files are created in the real cwd.
    Mirrors the real layout: <skills-path>/my-skill/SKILL.md
    """
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: A test skill.\n---\nBody content.",
    )
    monkeypatch.setitem(SKILLS_PATHS, "project", [tmp_path])
    monkeypatch.setitem(SKILLS_PATHS, "user", [])
    return tmp_path


def test_discover_skills_skips_nonexistent_paths() -> None:
    """Tests discover_skills skips scope paths that do not exist."""
    skills = discover_skills(Path("."))

    assert skills == []


def test_discover_skills_skips_non_directories(tmp_path: Path) -> None:
    """Tests discover_skills skips non-directory entries in skills path."""
    (tmp_path / "not_a_dir.txt").write_text("hello")
    fake_paths: dict[str, list[Path]] = {"project": [tmp_path], "user": []}
    with (
        patch(
            "llm_agents_from_scratch.skills.discovery.SKILLS_PATHS",
            fake_paths,
        ),
        patch(
            "llm_agents_from_scratch.skills.discovery.validate_skill_dir",
        ) as mock_validate,
    ):
        skills = discover_skills(Path("."))

    mock_validate.assert_not_called()
    assert skills == []


def test_discover_skills_valid_skill_dir(skills_dir: Path) -> None:
    """Tests discover_skills creates a Skill for a valid skill directory."""
    mock_info = MagicMock()

    with (
        patch(
            "llm_agents_from_scratch.skills.discovery.validate_skill_dir",
            return_value=[],
        ),
        patch(
            "llm_agents_from_scratch.skills.discovery.SkillInfo.from_skill_md",
            return_value=mock_info,
        ),
    ):
        skills = discover_skills(Path("."))

    skill_dir = skills_dir / "my-skill"
    assert len(skills) == 1
    assert isinstance(skills[0], Skill)
    assert skills[0].scope == "project"
    assert skills[0].location == (skill_dir / "SKILL.md").resolve()


def test_discover_skills_emits_cosmetic_warnings(skills_dir: Path) -> None:
    """Tests discover_skills emits returned warnings and still loads skill."""
    assert skills_dir.exists()
    mock_info = MagicMock()
    cosmetic_warnings = [
        SkillValidationWarning("name does not match directory"),
    ]

    with (
        patch(
            "llm_agents_from_scratch.skills.discovery.validate_skill_dir",
            return_value=cosmetic_warnings,
        ),
        patch(
            "llm_agents_from_scratch.skills.discovery.SkillInfo.from_skill_md",
            return_value=mock_info,
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        skills = discover_skills(Path("."))

    assert len(skills) == 1
    assert len(caught) == 1
    assert issubclass(caught[0].category, SkillValidationWarning)
    assert "name does not match directory" in str(caught[0].message)


def test_discover_skills_skips_on_fatal_error(skills_dir: Path) -> None:
    """Tests discover_skills emits SkillSkippedWarning and skips the skill."""
    assert skills_dir.exists()

    with (
        patch(
            "llm_agents_from_scratch.skills.discovery.validate_skill_dir",
            side_effect=SkillValidationError("missing SKILL.md"),
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        skills = discover_skills(Path("."))

    assert skills == []
    assert len(caught) == 1
    assert issubclass(caught[0].category, SkillSkippedWarning)
    assert "missing SKILL.md" in str(caught[0].message)
