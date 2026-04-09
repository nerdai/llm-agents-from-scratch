"""Unit tests for Skill."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_agents_from_scratch.data_structures.skill import (
    SkillFrontmatter,
    SkillScope,
)
from llm_agents_from_scratch.errors import (
    EmptySkillBodyError,
    InvalidFrontmatterError,
    SkillValidationError,
)
from llm_agents_from_scratch.skills.skill import Skill


def make_skill(scope: SkillScope = SkillScope.PROJECT) -> Skill:
    """Create a Skill instance for testing."""
    info = MagicMock()
    location = Path("/fake/skill/SKILL.md")
    return Skill(frontmatter=info, location=location, scope=scope)  # type: ignore[arg-type]


def test_skill_init_sets_attributes() -> None:
    """Tests Skill.__init__ sets info, location, and scope."""
    info = MagicMock()
    location = Path("/fake/skill/SKILL.md")
    skill = Skill(frontmatter=info, location=location, scope=SkillScope.PROJECT)

    assert skill.frontmatter is info
    assert skill.location == location
    assert skill.scope == SkillScope.PROJECT


def test_skill_init_user_scope() -> None:
    """Tests Skill.__init__ accepts user scope."""
    skill = make_skill(scope=SkillScope.USER)
    assert skill.scope == SkillScope.USER


def test_skill_read_body_returns_stripped_body(tmp_path: Path) -> None:
    """Tests Skill.read_body() returns body without frontmatter."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\nname: my-skill\ndescription: Does things.\n---\n\n"
        "## Instructions\n\nDo the thing.\n",
    )
    info = SkillFrontmatter(name="my-skill", description="Does things.")
    skill = Skill(frontmatter=info, location=skill_md, scope=SkillScope.PROJECT)

    assert skill.read_body() == "## Instructions\n\nDo the thing."


def test_skill_read_body_raises_on_empty_body(tmp_path: Path) -> None:
    """Tests Skill.read_body() raises EmptySkillBodyError for empty body."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text("---\nname: my-skill\ndescription: Does things.\n---\n")
    info = SkillFrontmatter(name="my-skill", description="Does things.")
    skill = Skill(frontmatter=info, location=skill_md, scope=SkillScope.PROJECT)

    with pytest.raises(EmptySkillBodyError):
        skill.read_body()


def test_skill_read_body_raises_on_unreadable_file(tmp_path: Path) -> None:
    """Tests Skill.read_body() raises SkillValidationError if unreadable."""
    skill_md = tmp_path / "SKILL.md"
    info = SkillFrontmatter(name="my-skill", description="Does things.")
    skill = Skill(frontmatter=info, location=skill_md, scope=SkillScope.PROJECT)

    with pytest.raises(SkillValidationError):
        skill.read_body()


def test_skill_read_body_raises_on_missing_delimiters(tmp_path: Path) -> None:
    """Tests Skill.read_body() raises InvalidFrontmatterError if no delims."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text("name: my-skill\ndescription: Does things.\n")
    info = SkillFrontmatter(name="my-skill", description="Does things.")
    skill = Skill(frontmatter=info, location=skill_md, scope=SkillScope.PROJECT)

    with pytest.raises(InvalidFrontmatterError):
        skill.read_body()


def test_skill_resources_returns_empty_when_no_subdirs(tmp_path: Path) -> None:
    """Tests Skill.resources returns [] when no optional subdirs exist."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\nname: my-skill\ndescription: Does things.\n---\n\nBody.\n",
    )
    info = SkillFrontmatter(name="my-skill", description="Does things.")
    skill = Skill(frontmatter=info, location=skill_md, scope=SkillScope.PROJECT)

    assert skill.resources == []


def test_skill_resources_returns_relative_paths(tmp_path: Path) -> None:
    """Tests Skill.resources returns relative paths for files in subdirs."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\nname: my-skill\ndescription: Does things.\n---\n\nBody.\n",
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("print('hi')")
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "guide.md").write_text("# Guide")
    info = SkillFrontmatter(name="my-skill", description="Does things.")
    skill = Skill(frontmatter=info, location=skill_md, scope=SkillScope.PROJECT)

    resources = skill.resources
    assert Path("scripts/run.py") in resources
    assert Path("references/guide.md") in resources


def test_skill_catalog_returns_xml_snippet() -> None:
    """Tests Skill.catalog() returns expected XML string."""
    info = SkillFrontmatter(
        name="pdf-processing",
        description="Handle PDF files.",
    )
    location = Path("/home/user/.agents/skills/pdf-processing/SKILL.md")
    skill = Skill(frontmatter=info, location=location, scope=SkillScope.PROJECT)

    expected = (
        "<skill>\n"
        "    <name>pdf-processing</name>\n"
        "    <description>Handle PDF files.</description>\n"
        "    <location>"
        "/home/user/.agents/skills/pdf-processing/SKILL.md"
        "</location>\n"
        "  </skill>"
    )
    assert skill.catalog() == expected
