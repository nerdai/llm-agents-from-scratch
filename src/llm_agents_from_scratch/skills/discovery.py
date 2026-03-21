"""Functions for skill discovery."""

import warnings
from pathlib import Path

from ..data_structures.skill import SkillInfo
from ..errors import SkillValidationError, SkillValidationWarning
from .constants import SKILLS_PATHS
from .skill import Skill


def validate_skill_dir(dir: Path) -> list[SkillValidationError]:
    """Validate a directory as a skill directory.

    A valid skill directory must contain a SKILL.md file with a valid
    name and description in its frontmatter.

    Args:
        dir: Path to the directory to validate.

    Returns:
        List of validation errors. Empty list means the directory is valid.
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
                errors = validate_skill_dir(skill_dir)
                if not errors:
                    with open(skill_dir / "SKILL.md", "r") as f:
                        skill_md = f.read()
                        info = SkillInfo.from_skill_md(skill_md)
                        skills.append(
                            Skill(
                                info=info,
                                location=(skill_dir / "SKILL.md").resolve(),
                                scope=scope,  # type: ignore[arg-type]
                            ),
                        )
                else:
                    for error in errors:
                        warnings.warn(
                            str(error),
                            SkillValidationWarning,
                            stacklevel=2,
                        )

    return skills
