"""Utility functions for skills."""

from pathlib import Path

from ..data_structures.skill import SkillScope

_SKILL_SUBDIRS = [".from_scratch/skills", ".agent/skills"]


def get_skills_paths(scope: SkillScope) -> list[Path]:
    """Return skill directories to scan for a given scope.

    Paths are resolved at call time so that changes to the working directory
    between import and discovery are correctly reflected.

    Args:
        scope: The skill scope to resolve paths for.

    Returns:
        List of paths to scan for skills.
    """
    base = Path.cwd() if scope == SkillScope.PROJECT else Path.home()
    return [base / subdir for subdir in _SKILL_SUBDIRS]
