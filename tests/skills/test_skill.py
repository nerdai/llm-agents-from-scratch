"""Unit tests for Skill."""

from pathlib import Path
from unittest.mock import MagicMock

from llm_agents_from_scratch.data_structures.skill import SkillInfo
from llm_agents_from_scratch.skills.constants import CATALOG_SKILL_TEMPLATE
from llm_agents_from_scratch.skills.skill import Skill


def make_skill(scope: str = "project") -> Skill:
    """Create a Skill instance for testing."""
    info = MagicMock()
    location = Path("/fake/skill/SKILL.md")
    return Skill(info=info, location=location, scope=scope)  # type: ignore[arg-type]


def test_skill_init_sets_attributes() -> None:
    """Tests Skill.__init__ sets info, location, and scope."""
    info = MagicMock()
    location = Path("/fake/skill/SKILL.md")
    skill = Skill(info=info, location=location, scope="project")  # type: ignore[arg-type]

    assert skill.info is info
    assert skill.location == location
    assert skill.scope == "project"


def test_skill_init_user_scope() -> None:
    """Tests Skill.__init__ accepts user scope."""
    skill = make_skill(scope="user")
    assert skill.scope == "user"


def test_skill_catalog_returns_xml_snippet() -> None:
    """Tests Skill.catalog() returns expected XML string."""
    info = SkillInfo(name="pdf-processing", description="Handle PDF files.")
    location = Path("/home/user/.agents/skills/pdf-processing/SKILL.md")
    skill = Skill(info=info, location=location, scope="project")

    expected = CATALOG_SKILL_TEMPLATE.format(
        name="pdf-processing",
        description="Handle PDF files.",
        location=location.as_posix(),
    )
    assert skill.catalog() == expected
