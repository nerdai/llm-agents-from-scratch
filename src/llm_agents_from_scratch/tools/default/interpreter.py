"""PythonInterpreterTool default tool."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from ...base.tool import BaseTool
from ...data_structures import ToolCall, ToolCallResult


class PythonInterpreterTool(BaseTool):
    """A tool for executing a local Python script."""

    @property
    def name(self) -> str:
        """Name of the Python interpreter tool."""
        return "from_scratch__python_interpreter"

    @property
    def description(self) -> str:
        """Description of the Python interpreter tool."""
        return (
            "Execute a local Python script and return its stdout output. "
            "Optionally pipe text to the script via stdin."
        )

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the Python script to execute.",
                },
                "stdin": {
                    "type": "string",
                    "description": (
                        "Optional text to pass to the script via stdin."
                    ),
                },
            },
            "required": ["path"],
        }

    def __call__(
        self,
        tool_call: ToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ToolCallResult:
        """Execute the Python interpreter tool.

        Args:
            tool_call (ToolCall): The ToolCall to execute.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ToolCallResult: The script stdout, or an error result.
        """
        raw_path = tool_call.arguments.get("path")
        if not raw_path:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=json.dumps(
                    {"error_type": "ValueError", "message": "Missing 'path'."},
                ),
                error=True,
            )

        if not Path(raw_path).exists():
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=json.dumps(
                    {
                        "error_type": "FileNotFoundError",
                        "message": f"Script not found: '{raw_path}'.",
                    },
                ),
                error=True,
            )

        stdin_text = tool_call.arguments.get("stdin")
        result = subprocess.run(
            [sys.executable, raw_path],
            check=False,
            capture_output=True,
            text=True,
            input=stdin_text,
        )

        if result.returncode != 0:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=json.dumps(
                    {
                        "error_type": "RuntimeError",
                        "message": result.stderr,
                    },
                ),
                error=True,
            )

        return ToolCallResult(
            tool_call_id=tool_call.id_,
            content=result.stdout,
            error=False,
        )
