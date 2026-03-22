"""Skill construct."""

from pathlib import Path

from ..data_structures.skill import SkillInfo, SkillScope
from ..errors import (
    EmptySkillBodyError,
    InvalidFrontmatterError,
    SkillValidationError,
)
from .constants import CATALOG_SKILL_TEMPLATE


class Skill:
    """A skill that can be discovered and activated by an LLM agent.

    Attributes:
        info: Parsed frontmatter metadata from the skill's SKILL.md file.
        location: Absolute path to the skill's SKILL.md file on disk.
        scope: The scope of the skill (`SkillScope.PROJECT` or
            `SkillScope.USER`). Used to resolve name collisions via
            deterministic precedence, with `SkillScope.PROJECT` taking
            priority over `SkillScope.USER`.
    """

    def __init__(
        self,
        info: SkillInfo,
        location: Path,
        scope: SkillScope,
    ):
        """Instantiate a Skill.

        Args:
            info: Parsed frontmatter metadata from the skill's SKILL.md file.
            location: Absolute path to the skill's SKILL.md file on disk.
            scope: The scope of the skill (`SkillScope.PROJECT` or
                `SkillScope.USER`).
        """
        self.info = info
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

    def catalog(self) -> str:
        """Returns XML structured string for cataloging skill."""
        return CATALOG_SKILL_TEMPLATE.format(
            name=self.info.name,
            description=self.info.description,
            location=self.location.as_posix(),
        )
