"""Unit tests for UseSkillTool."""

from pathlib import Path

from llm_agents_from_scratch.data_structures import ToolCall
from llm_agents_from_scratch.data_structures.skill import (
    SkillFrontmatter,
    SkillScope,
)
from llm_agents_from_scratch.skills.skill import Skill
from llm_agents_from_scratch.skills.tools import UseSkillTool


def make_skill(
    name: str = "my-skill",
    description: str = "Does things.",
    disable_model_invocation: bool = False,
    location: Path = Path("/fake/skill/SKILL.md"),
    scope: SkillScope = SkillScope.PROJECT,
) -> Skill:
    info = SkillFrontmatter(name=name, description=description)
    return Skill(
        frontmatter=info,
        location=location,
        scope=scope,
        disable_model_invocation=disable_model_invocation,
    )


def test_use_skill_tool_name() -> None:
    """Tests UseSkillTool.name returns 'use_skill'."""
    tool = UseSkillTool(skills={})
    assert tool.name == "from_scratch__use_skill"


def test_use_skill_tool_description() -> None:
    """Tests UseSkillTool.description mentions activation."""
    tool = UseSkillTool(skills={})
    assert "activate" in tool.description.lower()


def test_use_skill_tool_parameters_json_schema_enum() -> None:
    """Tests parameters_json_schema enum contains visible skill names."""
    skills = {
        "skill-a": make_skill(name="skill-a"),
        "skill-b": make_skill(name="skill-b"),
    }
    tool = UseSkillTool(skills=skills)

    enum = tool.parameters_json_schema["properties"]["name"]["enum"]
    assert "skill-a" in enum
    assert "skill-b" in enum


def test_use_skill_tool_parameters_json_schema_excludes_disabled() -> None:
    """Tests parameters_json_schema enum excludes disable-model-invocation."""
    skills = {
        "visible": make_skill(name="visible"),
        "hidden": make_skill(name="hidden", disable_model_invocation=True),
    }
    tool = UseSkillTool(skills=skills)

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
        frontmatter=SkillFrontmatter(
            name="my-skill",
            description="Does things.",
        ),
        location=skill_md,
        scope=SkillScope.PROJECT,
    )
    tool = UseSkillTool(skills={"my-skill": skill})

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
        frontmatter=SkillFrontmatter(
            name="my-skill",
            description="Does things.",
        ),
        location=skill_md,
        scope=SkillScope.PROJECT,
    )
    tool = UseSkillTool(skills={"my-skill": skill})

    content = tool._build_skill_content("my-skill")

    assert "<skill_resources>" in content
    assert "<file>scripts/run.py</file>" in content
    assert (
        "Relative paths in this skill are relative to the skill directory."
        in content
    )


def test_use_skill_tool_call_returns_skill_content(tmp_path: Path) -> None:
    """Tests __call__ returns skill content on successful activation."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\nname: my-skill\ndescription: Does things.\n---\n\n"
        "## Instructions\n\nDo the thing.\n",
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run.py").write_text("print('hi')")
    (tmp_path / "references").mkdir()
    (tmp_path / "references" / "guide.md").write_text("# Guide")
    skill = Skill(
        frontmatter=SkillFrontmatter(
            name="my-skill",
            description="Does things.",
        ),
        location=skill_md,
        scope=SkillScope.PROJECT,
    )
    tool = UseSkillTool(skills={"my-skill": skill})
    tool_call = ToolCall(tool_name="use_skill", arguments={"name": "my-skill"})

    result = tool(tool_call=tool_call)

    assert result.error is False
    assert '<skill_content name="my-skill">' in result.content  # type: ignore
    assert "## Instructions" in result.content  # type: ignore
    assert "<skill_resources>" in result.content  # type: ignore
    assert "<file>scripts/run.py</file>" in result.content  # type: ignore
    assert "<file>references/guide.md</file>" in result.content  # type: ignore


def test_use_skill_tool_call_returns_error_on_invalid_name(
    tmp_path: Path,
) -> None:
    """Tests __call__ returns error ToolCallResult for unknown skill name."""
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\nname: my-skill\ndescription: Does things.\n---\n\nBody.\n",
    )
    skill = Skill(
        frontmatter=SkillFrontmatter(
            name="my-skill",
            description="Does things.",
        ),
        location=skill_md,
        scope=SkillScope.PROJECT,
    )
    tool = UseSkillTool(skills={"my-skill": skill})
    tool_call = ToolCall(
        tool_name="use_skill",
        arguments={"name": "nonexistent"},
    )

    result = tool(tool_call=tool_call)

    assert result.error is True


def test_use_skill_tool_call_returns_error_when_skill_missing_from_registry(
    tmp_path: Path,
) -> None:
    """Tests __call__ guard when skill passes enum but is absent from registry.

    Simulates a case where _skills and _visible diverge — the guard in
    __call__ catches this before _build_skill_content raises a KeyError.
    run_with_skill() piggybacks on this guard for unknown skill names.
    """
    skill_md = tmp_path / "SKILL.md"
    skill_md.write_text(
        "---\nname: my-skill\ndescription: Does things.\n---\n\nBody.\n",
    )
    skill = Skill(
        frontmatter=SkillFrontmatter(
            name="my-skill",
            description="Does things.",
        ),
        location=skill_md,
        scope=SkillScope.PROJECT,
    )
    tool = UseSkillTool(skills={"my-skill": skill})
    # simulate divergence: name still in enum (_visible) but removed
    # from registry
    del tool._skills["my-skill"]
    tool_call = ToolCall(
        tool_name="use_skill",
        arguments={"name": "my-skill"},
    )

    result = tool(tool_call=tool_call)

    assert result.error is True
    assert "not found" in result.content  # type: ignore
