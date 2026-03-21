"""Data Structures for Skills."""

from pydantic import BaseModel, ConfigDict, Field


class SkillInfo(BaseModel):
    """Parsed frontmatter metadata from a skill's SKILL.md file.

    Attributes:
        name: Name of the skill.
        description: Description of what the skill does and when to use it.
        license: License name or reference to a bundled license file.
        compatibility: Environment requirements for the skill.
        metadata: Arbitrary key-value pairs for additional metadata.
        allowed_tools: Space-delimited list of pre-approved tools the skill
            may use.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] | None = None
    allowed_tools: str | None = Field(None, alias="allowed-tools")

    @classmethod
    def from_skill_md(cls, skill_md: str) -> "SkillInfo":
        """Construct from a loaded SKILL.md file."""
        raise NotImplementedError  # pragma: no cover
