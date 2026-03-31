"""Tools for Skills activation."""

import json
from typing import Any

from jsonschema import SchemaError, ValidationError, validate

from ..base.tool import BaseTool
from ..data_structures import ToolCall, ToolCallResult
from .constants import ACTIVATION_CONTENT_TEMPLATE, SKILL_RESOURCES_TEMPLATE
from .skill import Skill


class UseSkillTool(BaseTool):
    """A dedicated tool for activating a skill.

    Attributes:
        skills (dict[str, Skill]): All discovered skills, keyed by name.
            Includes skills with ``disable-model-invocation`` set — those are
            excluded from the enum but remain loadable if called directly.
    """

    def __init__(
        self,
        skills: dict[str, Skill],
    ) -> None:
        """Initialize a UseSkillTool.

        Args:
            skills (dict[str, Skill]): All discovered skills, keyed by name.
        """
        self._skills = skills
        self._visible = [
            name
            for name, s in skills.items()
            if not s.info.disable_model_invocation
        ]

    @property
    def name(self) -> str:
        """Name of skill activation tool."""
        return "from_scratch__use_skill"

    @property
    def description(self) -> str:
        """Description of the skill activation tool."""
        return (
            "Load and activate a skill by name, returning its full"
            " instructions. Only call this tool with a skill name from"
            " the <available_skills> catalog."
        )

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters.

        The ``name`` field is constrained to an enum of visible skill names
        (i.e. skills that have not set ``disable-model-invocation``).
        """
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": self._visible},
            },
            "required": ["name"],
        }

    def _build_skill_content(self, name: str) -> str:
        """Build the ``<skill_content>`` block for a skill.

        Args:
            name: The skill name to build content for.

        Returns:
            Formatted ``<skill_content>`` XML string.
        """
        skill = self._skills[name]
        resources = skill.resources
        skill_resources = (
            SKILL_RESOURCES_TEMPLATE.format(
                files="\n".join(
                    f"  <file>{r.as_posix()}</file>" for r in resources
                ),
            )
            if resources
            else ""
        )
        return ACTIVATION_CONTENT_TEMPLATE.format(
            name=skill.info.name,
            body=skill.read_body(),
            skill_dir=skill.location.parent.as_posix(),
            skill_resources=skill_resources,
        )

    def __call__(
        self,
        tool_call: ToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ToolCallResult:
        """Execute the skill activation tool.

        Args:
            tool_call (ToolCall): The ToolCall to execute.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ToolCallResult: The activated skill's full content.
        """
        try:
            # validate the arguments
            validate(tool_call.arguments, schema=self.parameters_json_schema)
        except (SchemaError, ValidationError) as e:
            error_details = {
                "error_type": e.__class__.__name__,
                "message": e.message,
            }
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=json.dumps(error_details),
                error=True,
            )

        # if pass validation then "name" is present in arguments
        skill_name: str = tool_call.arguments["name"]
        if skill_name not in self._skills:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=json.dumps(
                    {
                        "error_type": "ValueError",
                        "message": f"Skill '{skill_name}' not found.",
                    },
                ),
                error=True,
            )
        content = self._build_skill_content(skill_name)

        return ToolCallResult(
            tool_call_id=tool_call.id_,
            content=content,
            error=False,
        )
