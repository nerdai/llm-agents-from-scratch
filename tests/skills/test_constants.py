"""Unit tests for skills constants."""

from pathlib import Path

from llm_agents_from_scratch.skills.constants import SKILLS_PATHS


def test_skills_paths_has_project_and_user_scopes() -> None:
    """Tests SKILLS_PATHS has both project and user scopes."""
    assert "project" in SKILLS_PATHS
    assert "user" in SKILLS_PATHS


def test_skills_paths_values_are_lists_of_paths() -> None:
    """Tests SKILLS_PATHS values are lists of Path objects."""
    for paths in SKILLS_PATHS.values():
        assert isinstance(paths, list)
        for path in paths:
            assert isinstance(path, Path)


def test_skills_paths_project_scope_is_cwd_relative() -> None:
    """Tests project scope paths are relative to cwd."""
    for path in SKILLS_PATHS["project"]:
        assert path.is_absolute()
        assert str(path).startswith(str(Path.cwd()))


def test_skills_paths_user_scope_is_home_relative() -> None:
    """Tests user scope paths are relative to home directory."""
    for path in SKILLS_PATHS["user"]:
        assert path.is_absolute()
        assert str(path).startswith(str(Path.home()))
