"""Unit tests for Skill."""

from pathlib import Path
from unittest.mock import MagicMock

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
