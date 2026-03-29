"""Unit tests for default tools."""

from pathlib import Path
from unittest.mock import patch

from llm_agents_from_scratch.data_structures import ToolCall
from llm_agents_from_scratch.tools.default import (
    DEFAULT_TOOLS,
    PythonInterpreterTool,
    ReadFileTool,
)


def test_read_file_tool_name() -> None:
    """Tests ReadFileTool.name."""
    tool = ReadFileTool()
    assert tool.name == "from_scratch__read_file"


def test_read_file_tool_description() -> None:
    """Tests ReadFileTool.description mentions reading."""
    tool = ReadFileTool()
    assert "read" in tool.description.lower()


def test_read_file_tool_parameters_json_schema() -> None:
    """Tests ReadFileTool.parameters_json_schema has required path field."""
    tool = ReadFileTool()
    schema = tool.parameters_json_schema
    assert schema["type"] == "object"
    assert "path" in schema["properties"]
    assert schema["required"] == ["path"]


def test_read_file_tool_reads_file(tmp_path: Path) -> None:
    """Tests ReadFileTool returns file contents."""
    f = tmp_path / "hello.txt"
    f.write_text("hello world")
    tool = ReadFileTool()
    tool_call = ToolCall(
        tool_name="from_scratch__read_file",
        arguments={"path": str(f)},
    )

    result = tool(tool_call=tool_call)

    assert result.error is False
    assert result.content == "hello world"


def test_read_file_tool_error_on_missing_file(tmp_path: Path) -> None:
    """Tests ReadFileTool returns error for non-existent file."""
    tool = ReadFileTool()
    tool_call = ToolCall(
        tool_name="from_scratch__read_file",
        arguments={"path": str(tmp_path / "nonexistent.txt")},
    )

    result = tool(tool_call=tool_call)

    assert result.error is True
    assert "FileNotFoundError" in result.content


def test_read_file_tool_error_on_missing_path_argument() -> None:
    """Tests ReadFileTool returns error when path argument is absent."""
    tool = ReadFileTool()
    tool_call = ToolCall(
        tool_name="from_scratch__read_file",
        arguments={},
    )

    result = tool(tool_call=tool_call)

    assert result.error is True
    assert "Missing" in result.content


def test_read_file_tool_error_on_os_error(tmp_path: Path) -> None:
    """Tests ReadFileTool returns error on generic OSError."""
    f = tmp_path / "file.txt"
    f.write_text("data")

    tool = ReadFileTool()
    tool_call = ToolCall(
        tool_name="from_scratch__read_file",
        arguments={"path": str(f)},
    )

    with patch.object(
        Path,
        "read_text",
        side_effect=OSError("permission denied"),
    ):
        result = tool(tool_call=tool_call)

    assert result.error is True
    assert "OSError" in result.content


def test_default_tools_contains_read_file_tool() -> None:
    """Tests DEFAULT_TOOLS includes a ReadFileTool instance."""
    assert any(isinstance(t, ReadFileTool) for t in DEFAULT_TOOLS)


# --- PythonInterpreterTool ---


def test_python_interpreter_tool_name() -> None:
    """Tests PythonInterpreterTool.name."""
    tool = PythonInterpreterTool()
    assert tool.name == "from_scratch__python_interpreter"


def test_python_interpreter_tool_description() -> None:
    """Tests PythonInterpreterTool.description mentions executing."""
    tool = PythonInterpreterTool()
    assert "execut" in tool.description.lower()


def test_python_interpreter_tool_parameters_json_schema() -> None:
    """Tests PythonInterpreterTool schema has required path field."""
    tool = PythonInterpreterTool()
    schema = tool.parameters_json_schema
    assert schema["type"] == "object"
    assert "path" in schema["properties"]
    assert schema["required"] == ["path"]


def test_python_interpreter_tool_runs_script(tmp_path: Path) -> None:
    """Tests PythonInterpreterTool returns stdout of a script."""
    script = tmp_path / "hello.py"
    script.write_text('print("hello from script")')
    tool = PythonInterpreterTool()
    tool_call = ToolCall(
        tool_name="from_scratch__python_interpreter",
        arguments={"path": str(script)},
    )

    result = tool(tool_call=tool_call)

    assert result.error is False
    assert "hello from script" in result.content


def test_python_interpreter_tool_error_on_missing_script(
    tmp_path: Path,
) -> None:
    """Tests PythonInterpreterTool returns error for non-existent script."""
    tool = PythonInterpreterTool()
    tool_call = ToolCall(
        tool_name="from_scratch__python_interpreter",
        arguments={"path": str(tmp_path / "missing.py")},
    )

    result = tool(tool_call=tool_call)

    assert result.error is True
    assert "FileNotFoundError" in result.content


def test_python_interpreter_tool_error_on_missing_path_argument() -> None:
    """Tests PythonInterpreterTool returns error when path is absent."""
    tool = PythonInterpreterTool()
    tool_call = ToolCall(
        tool_name="from_scratch__python_interpreter",
        arguments={},
    )

    result = tool(tool_call=tool_call)

    assert result.error is True
    assert "Missing" in result.content


def test_python_interpreter_tool_error_on_script_failure(
    tmp_path: Path,
) -> None:
    """Tests PythonInterpreterTool returns error when script raises."""
    script = tmp_path / "bad.py"
    script.write_text("raise ValueError('oops')")
    tool = PythonInterpreterTool()
    tool_call = ToolCall(
        tool_name="from_scratch__python_interpreter",
        arguments={"path": str(script)},
    )

    result = tool(tool_call=tool_call)

    assert result.error is True
    assert "RuntimeError" in result.content


def test_default_tools_contains_python_interpreter_tool() -> None:
    """Tests DEFAULT_TOOLS includes a PythonInterpreterTool instance."""
    assert any(isinstance(t, PythonInterpreterTool) for t in DEFAULT_TOOLS)
