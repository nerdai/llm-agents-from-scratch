"""Functions for skill discovery."""

import warnings
from pathlib import Path

import yaml
from pydantic import ValidationError

from ..data_structures.skill import SkillFrontmatter, SkillScope
from ..errors import (
    EmptySkillBodyError,
    InvalidFrontmatterError,
    MissingSkillMdError,
    NameMismatchWarning,
    NameTooLongWarning,
    SkillShadowedWarning,
    SkillSkippedWarning,
    SkillValidationError,
    SkillValidationWarning,
)
from .constants import MAX_NAME_LENGTH
from .skill import Skill
from .utils import get_skills_paths


def validate_skill_dir(
    dir: Path,
) -> tuple[SkillFrontmatter, list[SkillValidationWarning]]:
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
        Tuple[SkillFrontmatter, list[SkillValidationWarning]]: A pair where
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
        raise MissingSkillMdError(
            f"Missing SKILL.md file in skill directory: {dir}",
        )

    try:
        with open(skill_md_path, "r", encoding="utf-8") as f:
            skill_md = f.read()
    except OSError as e:
        raise SkillValidationError(
            f"Failed to read SKILL.md at {skill_md_path}: {e}",
        ) from e

    try:
        _, frontmatter_str, body = skill_md.split("---", 2)
        frontmatter = yaml.safe_load(frontmatter_str)
        info = SkillFrontmatter.model_validate(frontmatter)
    except (ValueError, ValidationError, yaml.YAMLError) as e:
        raise InvalidFrontmatterError(str(e)) from e

    if not body.strip():
        raise EmptySkillBodyError

    if info.name != dir.name:
        skill_warnings.append(
            NameMismatchWarning(
                f"Skill name '{info.name}' does not match "
                f"directory name '{dir.name}'.",
            ),
        )

    if len(info.name) > MAX_NAME_LENGTH:
        skill_warnings.append(
            NameTooLongWarning(
                f"Skill name '{info.name}' is {len(info.name)} characters"
                f" long, which exceeds the maximum allowed length of"
                f" {MAX_NAME_LENGTH}.",
            ),
        )

    return info, skill_warnings


def discover_skills(scopes: list[SkillScope]) -> dict[str, Skill]:
    """Scan directories for skills across the provided scopes.

    Scopes are processed in the order given — on name collision, the last
    scope wins. To give `SkillScope.PROJECT` priority over `SkillScope.USER`,
    pass them as ``[SkillScope.USER, SkillScope.PROJECT]``.

    Args:
        scopes: The scopes to scan, in processing order.

    Returns:
        dict[str, Skill]: Discovered skills keyed by name, deduplicated with
            last-scope precedence.
    """
    skills: dict[str, Skill] = {}
    for scope in scopes:
        for skills_path in get_skills_paths(scope):
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

                if info.name in skills:
                    shadowed_scope = skills[info.name].scope
                    warnings.warn(
                        f"Skill '{info.name}' ({scope.value} scope) shadows"
                        f" an existing skill of the same name"
                        f" ({shadowed_scope.value} scope).",
                        SkillShadowedWarning,
                        stacklevel=2,
                    )
                skills[info.name] = Skill(
                    frontmatter=info,
                    location=(skill_dir / "SKILL.md").resolve(),
                    scope=scope,  # type: ignore[arg-type]
                )

    return skills
