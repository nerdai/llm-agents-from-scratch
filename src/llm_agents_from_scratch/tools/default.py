"""Default tools included with every LLMAgent."""

import json
from pathlib import Path
from typing import Any

from ..base.tool import BaseTool
from ..data_structures import ToolCall, ToolCallResult


class ReadFileTool(BaseTool):
    """A tool for reading the contents of a local file.

    Attributes:
        _base_dir (Path | None): Optional base directory used to resolve
            relative paths. When set, paths outside this directory are
            rejected.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """Initialize a ReadFileTool.

        Args:
            base_dir (Path | None, optional): Base directory to resolve
                relative paths against and enforce path containment.
                Defaults to None.
        """
        self._base_dir = base_dir

    @property
    def name(self) -> str:
        """Name of the read-file tool."""
        return "from_scratch__read_file"

    @property
    def description(self) -> str:
        """Description of the read-file tool."""
        return "Read the contents of a local file and return them as a string."

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read.",
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
        """Execute the read-file tool.

        Args:
            tool_call (ToolCall): The ToolCall to execute.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ToolCallResult: The file contents, or an error result.
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

        path = Path(raw_path)
        if self._base_dir is not None:
            path = (self._base_dir / path).resolve()
            try:
                path.relative_to(self._base_dir.resolve())
            except ValueError:
                return ToolCallResult(
                    tool_call_id=tool_call.id_,
                    content=json.dumps(
                        {
                            "error_type": "PermissionError",
                            "message": (
                                f"Path '{raw_path}' is outside the allowed"
                                " base directory."
                            ),
                        },
                    ),
                    error=True,
                )

        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=json.dumps(
                    {
                        "error_type": "FileNotFoundError",
                        "message": f"File not found: '{raw_path}'.",
                    },
                ),
                error=True,
            )
        except OSError as e:
            return ToolCallResult(
                tool_call_id=tool_call.id_,
                content=json.dumps(
                    {"error_type": "OSError", "message": str(e)},
                ),
                error=True,
            )

        return ToolCallResult(
            tool_call_id=tool_call.id_,
            content=content,
            error=False,
        )


DEFAULT_TOOLS: list[BaseTool] = [ReadFileTool()]
