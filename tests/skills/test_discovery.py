"""Unit tests for skill discovery."""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_agents_from_scratch.data_structures.skill import SkillScope
from llm_agents_from_scratch.errors import (
    SkillShadowedWarning,
    SkillSkippedWarning,
    SkillValidationError,
    SkillValidationWarning,
)
from llm_agents_from_scratch.skills.discovery import discover_skills
from llm_agents_from_scratch.skills.skill import Skill


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """Create a valid skill directory under tmp_path.

    Mirrors the real layout: <skills-path>/my-skill/SKILL.md
    """
    skill_dir = tmp_path / "my-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: A test skill.\n---\nBody content.",
    )
    return tmp_path


def _patch_paths(
    project: Path,
    user: Path | None = None,
):
    """Return a context manager that patches get_skills_path."""
    _user = user or Path("/nonexistent/__user__")
    paths = {SkillScope.PROJECT: project, SkillScope.USER: _user}
    return patch(
        "llm_agents_from_scratch.skills.discovery.get_skills_path",
        side_effect=lambda scope: paths[scope],
    )


def test_discover_skills_skips_nonexistent_paths() -> None:
    """Tests discover_skills skips scope paths that do not exist."""
    with _patch_paths(project=Path("/nonexistent/path")):
        skills = discover_skills(scopes=[SkillScope.PROJECT])

    assert skills == {}


def test_discover_skills_skips_non_directories(tmp_path: Path) -> None:
    """Tests discover_skills skips non-directory entries in skills path."""
    (tmp_path / "not_a_dir.txt").write_text("hello")

    with (
        _patch_paths(project=tmp_path),
        patch(
            "llm_agents_from_scratch.skills.discovery.validate_skill_dir",
        ) as mock_validate,
    ):
        skills = discover_skills(scopes=[SkillScope.PROJECT])

    mock_validate.assert_not_called()
    assert skills == {}


def test_discover_skills_valid_skill_dir(skills_dir: Path) -> None:
    """Tests discover_skills creates a Skill for a valid skill directory."""
    mock_info = MagicMock()
    mock_info.name = "my-skill"

    with (
        _patch_paths(project=skills_dir),
        patch(
            "llm_agents_from_scratch.skills.discovery.validate_skill_dir",
            return_value=(mock_info, []),
        ),
    ):
        skills = discover_skills(scopes=[SkillScope.PROJECT])

    skill_dir = skills_dir / "my-skill"
    assert len(skills) == 1
    assert isinstance(skills["my-skill"], Skill)
    assert skills["my-skill"].scope == SkillScope.PROJECT
    assert skills["my-skill"].location == (skill_dir / "SKILL.md").resolve()


def test_discover_skills_emits_cosmetic_warnings(skills_dir: Path) -> None:
    """Tests discover_skills emits returned warnings and still loads skill."""
    mock_info = MagicMock()
    mock_info.name = "my-skill"
    cosmetic_warnings = [
        SkillValidationWarning("name does not match directory"),
    ]

    with (
        _patch_paths(project=skills_dir),
        patch(
            "llm_agents_from_scratch.skills.discovery.validate_skill_dir",
            return_value=(mock_info, cosmetic_warnings),
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        skills = discover_skills(scopes=[SkillScope.PROJECT])

    assert len(skills) == 1
    assert len(caught) == 1
    assert issubclass(caught[0].category, SkillValidationWarning)
    assert "name does not match directory" in str(caught[0].message)


def test_discover_skills_skips_on_fatal_error(skills_dir: Path) -> None:
    """Tests discover_skills emits SkillSkippedWarning and skips the skill."""
    with (
        _patch_paths(project=skills_dir),
        patch(
            "llm_agents_from_scratch.skills.discovery.validate_skill_dir",
            side_effect=SkillValidationError("missing SKILL.md"),
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        skills = discover_skills(scopes=[SkillScope.PROJECT])

    assert skills == {}
    assert len(caught) == 1
    assert issubclass(caught[0].category, SkillSkippedWarning)
    assert "missing SKILL.md" in str(caught[0].message)


def test_discover_skills_project_overrides_user(tmp_path: Path) -> None:
    """Tests that project scope overwrites user scope on name collision."""
    user_dir = tmp_path / "user"
    project_dir = tmp_path / "project"
    for d in (user_dir, project_dir):
        skill_dir = d / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: A test skill.\n"
            "---\nBody content.",
        )

    user_info = MagicMock()
    user_info.name = "my-skill"
    project_info = MagicMock()
    project_info.name = "my-skill"

    def _validate(dir: Path):
        if dir.parent == user_dir:
            return user_info, []
        return project_info, []

    with (
        _patch_paths(project=project_dir, user=user_dir),
        patch(
            "llm_agents_from_scratch.skills.discovery.validate_skill_dir",
            side_effect=_validate,
        ),
    ):
        skills = discover_skills(
            scopes=[SkillScope.USER, SkillScope.PROJECT],
        )

    assert len(skills) == 1
    assert skills["my-skill"].scope == SkillScope.PROJECT


def test_discover_skills_emits_shadowed_warning(tmp_path: Path) -> None:
    """Tests that a SkillShadowedWarning is emitted on name collision."""
    user_dir = tmp_path / "user"
    project_dir = tmp_path / "project"
    for d in (user_dir, project_dir):
        skill_dir = d / "my-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: A test skill.\n"
            "---\nBody content.",
        )

    user_info = MagicMock()
    user_info.name = "my-skill"
    project_info = MagicMock()
    project_info.name = "my-skill"

    def _validate(dir: Path):
        if dir.parent == user_dir:
            return user_info, []
        return project_info, []

    with (
        _patch_paths(project=project_dir, user=user_dir),
        patch(
            "llm_agents_from_scratch.skills.discovery.validate_skill_dir",
            side_effect=_validate,
        ),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        discover_skills(scopes=[SkillScope.USER, SkillScope.PROJECT])

    shadowed = [
        w for w in caught if issubclass(w.category, SkillShadowedWarning)
    ]
    assert len(shadowed) == 1
    assert "my-skill" in str(shadowed[0].message)
    assert "project" in str(shadowed[0].message)
    assert "user" in str(shadowed[0].message)
