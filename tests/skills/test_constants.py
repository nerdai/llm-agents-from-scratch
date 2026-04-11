"""Unit tests for skills constants."""

from pathlib import Path

import pytest

from llm_agents_from_scratch.data_structures.skill import SkillScope
from llm_agents_from_scratch.skills.utils import get_skills_path


def test_get_skills_path_project_returns_cwd_relative_paths() -> None:
    """Tests project scope path is resolved relative to cwd."""
    path = get_skills_path(SkillScope.PROJECT)

    assert isinstance(path, Path)
    assert path.is_absolute()
    assert str(path).startswith(str(Path.cwd()))


def test_get_skills_path_user_returns_home_relative_paths() -> None:
    """Tests user scope path is resolved relative to home directory."""
    path = get_skills_path(SkillScope.USER)

    assert isinstance(path, Path)
    assert path.is_absolute()
    assert str(path).startswith(str(Path.home()))


def test_get_skills_path_project_reflects_cwd_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests project scope path reflects cwd at call time, not import time."""
    monkeypatch.chdir(tmp_path)
    path = get_skills_path(SkillScope.PROJECT)

    assert str(path).startswith(str(tmp_path))
