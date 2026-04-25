"""PythonInterpreterTool default tool."""

import json
import subprocess
import sys
import tempfile
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
            "Execute a Python script and return its stdout output. "
            "Provide either 'path' to run an existing script or 'code' to "
            "run inline code. Optionally pipe text to the script via stdin. "
            "Use 'cwd' to set the working directory — modules in that "
            "directory will be importable."
        )

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to an existing Python script.",
                },
                "code": {
                    "type": "string",
                    "description": (
                        "Inline Python code to execute. Use instead of 'path'."
                    ),
                },
                "stdin": {
                    "type": "string",
                    "description": (
                        "Optional text to pass to the script via stdin."
                    ),
                },
                "cwd": {
                    "type": "string",
                    "description": (
                        "Optional working directory for execution. "
                        "Modules inside this directory are importable."
                    ),
                },
            },
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
        code = tool_call.arguments.get("code")
        stdin_text = tool_call.arguments.get("stdin")
        raw_cwd = tool_call.arguments.get("cwd")

        if not raw_path and not code:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=json.dumps(
                    {
                        "error_type": "ValueError",
                        "message": "Missing 'path' or 'code'.",
                    },
                ),
                error=True,
            )

        cwd = Path(raw_cwd) if raw_cwd else None
        tmp_path: Path | None = None

        if code:
            tmp_dir = cwd or Path(tempfile.gettempdir())
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                dir=tmp_dir,
                delete=False,
            ) as tmp_file:
                tmp_file.write(code)
                tmp_path = Path(tmp_file.name)
            script_path: str = str(tmp_path)
        else:
            if not Path(raw_path).exists():  # type: ignore[arg-type]
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
            script_path = raw_path  # type: ignore[assignment]

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                check=False,
                capture_output=True,
                text=True,
                input=stdin_text,
                cwd=cwd,
            )
        finally:
            if tmp_path:
                tmp_path.unlink(missing_ok=True)

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
