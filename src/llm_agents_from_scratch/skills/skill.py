"""Skill construct."""

from pathlib import Path

from ..data_structures.skill import SkillFrontmatter, SkillScope
from ..errors import (
    EmptySkillBodyError,
    InvalidFrontmatterError,
    SkillValidationError,
)
from .constants import CATALOG_SKILL_TEMPLATE, OPTIONAL_SUBDIRS


class Skill:
    """A skill that can be discovered and activated by an LLM agent.

    Attributes:
        frontmatter: Parsed frontmatter metadata from the skill's SKILL.md file.
        location: Absolute path to the skill's SKILL.md file on disk.
        scope: The scope of the skill (`SkillScope.PROJECT` or
            `SkillScope.USER`). Used to resolve name collisions via
            deterministic precedence, with `SkillScope.PROJECT` taking
            priority over `SkillScope.USER`.
        resources: Relative paths of resource files found in the skill's
            optional subdirectories (``assets/``, ``scripts/``,
            ``references/``). Empty list if none of those directories exist.
    """

    def __init__(
        self,
        frontmatter: SkillFrontmatter,
        location: Path,
        scope: SkillScope,
    ):
        """Instantiate a Skill.

        Args:
            frontmatter: Parsed frontmatter metadata from the skill's
                SKILL.md file.
            location: Absolute path to the skill's SKILL.md file on disk.
            scope: The scope of the skill (`SkillScope.PROJECT` or
                `SkillScope.USER`).
        """
        self.frontmatter = frontmatter
        self.location = location
        self.scope = scope

    def read_body(self) -> str:
        """Return body content as string.

        Note: Most error conditions are caught during discovery. These errors
            can only be raised if the skill file is modified between discovery
            and activation.

        Raises:
            SkillValidationError: If the file cannot be read.
            InvalidFrontmatterError: If the frontmatter delimiters are missing
                or malformed.
            EmptySkillBodyError: If the body is empty or whitespace-only.
        """
        try:
            with open(self.location, "r", encoding="utf-8") as f:
                skill_md = f.read()
        except OSError as e:
            raise SkillValidationError(
                f"Failed to read SKILL.md at {self.location}: {e}",
            ) from e

        try:
            _, _, body = skill_md.split("---", 2)
        except ValueError as e:
            raise InvalidFrontmatterError(str(e)) from e

        if not body.strip():
            raise EmptySkillBodyError

        return body.strip()

    @property
    def resources(self) -> list[Path]:
        """Return relative paths of resource files in optional subdirectories.

        Scans the directories listed in ``OPTIONAL_SUBDIRS`` (``assets/``,
        ``scripts/``, ``references/``) under the skill directory and returns
        the relative path of each file found. Returns an empty list if none
        of those directories exist.

        Note: Only top-level files within each optional subdirectory are
            returned. This conforms to the Agent Skills specification, which
            advises keeping file references one level deep from ``SKILL.md``
            and avoiding deeply nested resource chains.
        """
        skill_dir = self.location.parent
        files: list[Path] = []
        for subdir in OPTIONAL_SUBDIRS:
            p = skill_dir / subdir
            if p.is_dir():
                files.extend(sorted(p.iterdir()))
        return [f.relative_to(skill_dir) for f in files if f.is_file()]

    def catalog(self) -> str:
        """Returns XML structured string for cataloging skill."""
        return CATALOG_SKILL_TEMPLATE.format(
            name=self.frontmatter.name,
            description=self.frontmatter.description,
            location=self.location.as_posix(),
        )
