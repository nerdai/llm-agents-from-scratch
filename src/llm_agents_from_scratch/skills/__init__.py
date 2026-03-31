"""Skills module."""

from ..data_structures.skill import SkillScope
from ..tools.default import PythonInterpreterTool, ReadFileTool
from .skill import Skill

# Tools for reading and executing files referenced in skill resources
# (assets/, scripts/, references/). Include these in LLMAgent when your
# skills make use of optional resource subdirectories.
TOOLS_FOR_SKILL_RESOURCES = [ReadFileTool(), PythonInterpreterTool()]

__all__ = ["Skill", "SkillScope", "TOOLS_FOR_SKILL_RESOURCES"]
