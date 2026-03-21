"""Functions for skill discovery."""

import warnings
from pathlib import Path

from ..data_structures.skill import SkillInfo
from ..errors import (
    SkillSkippedWarning,
    SkillValidationError,
    SkillValidationWarning,
)
from .constants import SKILLS_PATHS
from .skill import Skill


def validate_skill_dir(
    dir: Path,
) -> tuple[SkillInfo, list[SkillValidationWarning]]:
    """Validate a directory as a skill directory.

    A valid skill directory must contain a SKILL.md file with a valid
    name and description in its frontmatter.

    Cosmetic issues (e.g. name/directory mismatch) are returned as warnings
    so the skill is still loaded. Fatal issues (e.g. missing SKILL.md,
    unparseable frontmatter) are raised as errors so the caller can skip
    the skill and emit a SkillSkippedWarning.

    Args:
        dir: Path to the directory to validate.

    Returns:
        List of warnings for cosmetic issues. Empty list means no issues.

    Raises:
        SkillValidationError: If the skill directory has a fatal issue that
            prevents the skill from being loaded.
    """
    raise NotImplementedError  # pragma: no cover


def discover_skills(path: Path) -> list[Skill]:
    """Scan project and user directories for skills."""
    skills: list[Skill] = []
    for scope, skills_paths in SKILLS_PATHS.items():
        for skills_path in skills_paths:
            if not skills_path.exists():
                continue

            for skill_dir in sorted(skills_path.iterdir()):
                if not skill_dir.is_dir():
                    continue

                # validate dir is an actual Skill dir
                try:
                    info, skill_warnings = validate_skill_dir(skill_dir)
                except SkillValidationError as e:
                    # skip skill
                    warnings.warn(
                        str(e),
                        SkillSkippedWarning,
                        stacklevel=2,
                    )
                    continue

                for w in skill_warnings:
                    warnings.warn(str(w), type(w), stacklevel=2)
                skills.append(
                    Skill(
                        info=info,
                        location=(skill_dir / "SKILL.md").resolve(),
                        scope=scope,  # type: ignore[arg-type]
                    ),
                )

    return skills
