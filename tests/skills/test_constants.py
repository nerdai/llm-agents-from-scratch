"""Unit tests for skills constants."""

from pathlib import Path

import pytest

from llm_agents_from_scratch.data_structures.skill import SkillScope
from llm_agents_from_scratch.skills.utils import get_skills_paths


def test_get_skills_paths_project_returns_cwd_relative_paths() -> None:
    """Tests project scope paths are resolved relative to cwd."""
    paths = get_skills_paths(SkillScope.PROJECT)

    assert isinstance(paths, list)
    for path in paths:
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).startswith(str(Path.cwd()))


def test_get_skills_paths_user_returns_home_relative_paths() -> None:
    """Tests user scope paths are resolved relative to home directory."""
    paths = get_skills_paths(SkillScope.USER)

    assert isinstance(paths, list)
    for path in paths:
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).startswith(str(Path.home()))


def test_get_skills_paths_project_reflects_cwd_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests project scope paths reflect cwd at call time, not import time."""
    monkeypatch.chdir(tmp_path)
    paths = get_skills_paths(SkillScope.PROJECT)

    for path in paths:
        assert str(path).startswith(str(tmp_path))
