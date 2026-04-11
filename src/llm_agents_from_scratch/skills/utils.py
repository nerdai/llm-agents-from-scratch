"""Utility functions for skills."""

from pathlib import Path

from ..data_structures.skill import SkillScope

_SKILL_SUBDIR = ".agents/skills"


def get_skills_path(scope: SkillScope) -> Path:
    """Return skill directories to scan for a given scope.

    Paths are resolved at call time so that changes to the working directory
    between import and discovery are correctly reflected.

    Args:
        scope: The skill scope to resolve paths for.

    Returns:
        List of paths to scan for skills.
    """
    base = Path.cwd() if scope == SkillScope.PROJECT else Path.home()
    return base / _SKILL_SUBDIR
