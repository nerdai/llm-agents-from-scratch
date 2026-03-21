"""Functions for skill discovery."""

import warnings
from pathlib import Path

import yaml
from pydantic import ValidationError

from ..data_structures.skill import SkillInfo
from ..errors import (
    EmptySkillBodyError,
    InvalidFrontmatterError,
    MissingSkillMdError,
    NameMismatchWarning,
    NameTooLongWarning,
    SkillSkippedWarning,
    SkillValidationError,
    SkillValidationWarning,
)
from .constants import MAX_NAME_LENGTH, SKILLS_PATHS
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
        Tuple[SkillInfo, list[SkillValidationWarning]]: A pair where
            the first element is the validated skill metadata, and the
            second element is a list of warnings for cosmetic issues.
            An empty list means no issues.

    Raises:
        SkillValidationError: If the skill directory has a fatal issue that
            prevents the skill from being loaded.
    """
    skill_warnings: list[SkillValidationWarning] = []
    skill_md_path = dir / "SKILL.md"
    if not skill_md_path.is_file():
        raise MissingSkillMdError

    try:
        with open(skill_md_path, "r") as f:
            skill_md = f.read()

        _, frontmatter_str, body = skill_md.split("---", 2)
        frontmatter = yaml.safe_load(frontmatter_str)
        info = SkillInfo.model_validate(frontmatter)
    except (ValueError, ValidationError, yaml.YAMLError) as e:
        raise InvalidFrontmatterError(str(e)) from e

    if not body.strip():
        raise EmptySkillBodyError

    if info.name != dir.name:
        skill_warnings.append(NameMismatchWarning())

    if len(info.name) > MAX_NAME_LENGTH:
        skill_warnings.append(NameTooLongWarning())

    return info, skill_warnings


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
