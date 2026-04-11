"""Utility functions for skills."""

from pathlib import Path

from ..data_structures.skill import SkillScope
from .constants import SKILL_SUBDIR


def get_skills_path(scope: SkillScope) -> Path:
    """Return the skill directory to scan for a given scope.

    Paths are resolved at call time so that changes to the working directory
    between import and discovery are correctly reflected.

    Args:
        scope: The skill scope to resolve paths for.

    Returns:
        Path to scan for skills.
    """
    base = Path.cwd() if scope == SkillScope.PROJECT else Path.home()
    return base / SKILL_SUBDIR
