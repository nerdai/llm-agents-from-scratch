"""Skills module."""

from ..data_structures.skill import SkillScope
from ..tools.default import PythonInterpreterTool, ReadFileTool
from .skill import Skill

TOOLS_FOR_SKILL_RESOURCES: list = [ReadFileTool(), PythonInterpreterTool()]
"""Tools for reading and executing files referenced in skill resource subdirs.

Include these when building an agent whose skills use ``assets/``, ``scripts/``,
or ``references/`` subdirectories. Opt-in only — not added automatically.
"""

__all__ = ["Skill", "SkillScope", "TOOLS_FOR_SKILL_RESOURCES"]
