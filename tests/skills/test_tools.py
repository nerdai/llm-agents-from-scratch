"""Unit tests for UseSkillTool."""

from pathlib import Path

from llm_agents_from_scratch.data_structures.skill import SkillInfo, SkillScope
from llm_agents_from_scratch.skills.skill import Skill
from llm_agents_from_scratch.skills.tools import UseSkillTool


def make_skill(
    name: str = "my-skill",
    description: str = "Does things.",
    disable_model_invocation: bool = False,
    location: Path = Path("/fake/skill/SKILL.md"),
    scope: SkillScope = SkillScope.PROJECT,
) -> Skill:
    info = SkillInfo(
        name=name,
        description=description,
        **{"disable-model-invocation": disable_model_invocation},
    )
    return Skill(info=info, location=location, scope=scope)


def test_use_skill_tool_name() -> None:
    """Tests UseSkillTool.name returns 'use_skill'."""
    tool = UseSkillTool(skills={}, activated_skills=set())
    assert tool.name == "use_skill"


def test_use_skill_tool_description() -> None:
    """Tests UseSkillTool.description mentions activation."""
    tool = UseSkillTool(skills={}, activated_skills=set())
    assert "activate" in tool.description.lower()


def test_use_skill_tool_parameters_json_schema_enum() -> None:
    """Tests parameters_json_schema enum contains visible skill names."""
    skills = {
        "skill-a": make_skill(name="skill-a"),
        "skill-b": make_skill(name="skill-b"),
    }
    tool = UseSkillTool(skills=skills, activated_skills=set())

    enum = tool.parameters_json_schema["properties"]["name"]["enum"]
    assert "skill-a" in enum
    assert "skill-b" in enum


def test_use_skill_tool_parameters_json_schema_excludes_disabled() -> None:
    """Tests parameters_json_schema enum excludes disable-model-invocation."""
    skills = {
        "visible": make_skill(name="visible"),
        "hidden": make_skill(name="hidden", disable_model_invocation=True),
    }
    tool = UseSkillTool(skills=skills, activated_skills=set())

    enum = tool.parameters_json_schema["properties"]["name"]["enum"]
    assert "visible" in enum
    assert "hidden" not in enum


def test_use_skill_tool_build_skill_content_without_resources(
    tmp_path: Path,
) -> None:
    """Tests _build_skill_content returns correct XML with no resource dirs."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\nname: my-skill\ndescription: Does things.\n---\n\n"
        "## Instructions\n\nDo the thing.\n",
    )
    skill = Skill(
        info=SkillInfo(name="my-skill", description="Does things."),
        location=skill_md,
        scope=SkillScope.PROJECT,
    )
    tool = UseSkillTool(skills={"my-skill": skill}, activated_skills=set())

    content = tool._build_skill_content("my-skill")

    assert '<skill_content name="my-skill">' in content
    assert "## Instructions" in content
    assert f"Skill directory: {tmp_path.as_posix()}" in content
    assert "<skill_resources>" not in content
    assert content.endswith("</skill_content>")


def test_use_skill_tool_build_skill_content_with_resources(
    tmp_path: Path,
) -> None:
    """Tests _build_skill_content includes skill_resources when present."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\nname: my-skill\ndescription: Does things.\n---\n\n"
        "## Instructions\n\nDo the thing.\n",
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("print('hi')")
    skill = Skill(
        info=SkillInfo(name="my-skill", description="Does things."),
        location=skill_md,
        scope=SkillScope.PROJECT,
    )
    tool = UseSkillTool(skills={"my-skill": skill}, activated_skills=set())

    content = tool._build_skill_content("my-skill")

    assert "<skill_resources>" in content
    assert "<file>scripts/run.py</file>" in content
    assert (
        "Relative paths in this skill are relative to the skill directory."
        in content
    )
