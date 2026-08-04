"""Ready-to-use tools shipped with the framework, opt-in via `tools=...`."""

from ...base.tool import Tool
from .human_input import HumanInputTool, SharedConsoleHumanInputTool
from .interpreter import PythonInterpreterTool
from .read_file import ReadFileTool

DEFAULT_TOOLS: list[Tool] = [ReadFileTool(), PythonInterpreterTool()]

__all__ = [
    "DEFAULT_TOOLS",
    "HumanInputTool",
    "PythonInterpreterTool",
    "ReadFileTool",
    "SharedConsoleHumanInputTool",
]
