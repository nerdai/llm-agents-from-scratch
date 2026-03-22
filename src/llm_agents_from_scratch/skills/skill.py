"""Skill construct."""

from pathlib import Path
from typing import Literal

from ..data_structures.skill import SkillInfo
from ..errors import EmptySkillBodyError
from .constants import CATALOG_SKILL_TEMPLATE


class Skill:
    """A skill that can be discovered and activated by an LLM agent.

    Attributes:
        info: Parsed frontmatter metadata from the skill's SKILL.md file.
        location: Absolute path to the skill's SKILL.md file on disk.
        scope: The scope of the skill, either "project" or "user". Used to
            resolve name collisions via deterministic precedence, with
            "project" taking priority over "user".
    """

    def __init__(
        self,
        info: SkillInfo,
        location: Path,
        scope: Literal["project", "user"],
    ):
        """Instantiate a Skill.

        Args:
            info: Parsed frontmatter metadata from the skill's SKILL.md file.
            location: Absolute path to the skill's SKILL.md file on disk.
            scope: The scope of the skill, either "project" or "user".
        """
        self.info = info
        self.location = location
        self.scope = scope

    info: SkillInfo
    location: Path
    scope: Literal["project", "user"]

    def read_body(self) -> str:
        """Return body content as string.

        Note: Most error conditions (e.g. malformed frontmatter, missing
            delimiters) are caught during discovery. EmptySkillBodyError can
            only be raised if the skill file is modified between discovery
            and activation.
        """
        with open(self.location, "r") as f:
            skill_md = f.read()

        _, _, body = skill_md.split("---", 2)

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
