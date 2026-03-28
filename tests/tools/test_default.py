"""Unit tests for default tools."""

from pathlib import Path

from llm_agents_from_scratch.data_structures import ToolCall
from llm_agents_from_scratch.tools.default import DEFAULT_TOOLS, ReadFileTool


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


def test_read_file_tool_enforces_base_dir(tmp_path: Path) -> None:
    """Tests ReadFileTool rejects paths outside base_dir."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")

    tool = ReadFileTool(base_dir=allowed)
    tool_call = ToolCall(
        tool_name="from_scratch__read_file",
        arguments={"path": str(outside)},
    )

    result = tool(tool_call=tool_call)

    assert result.error is True
    assert "PermissionError" in result.content


def test_read_file_tool_allows_path_within_base_dir(tmp_path: Path) -> None:
    """Tests ReadFileTool reads files within base_dir."""
    base = tmp_path / "base"
    base.mkdir()
    f = base / "data.txt"
    f.write_text("safe content")

    tool = ReadFileTool(base_dir=base)
    tool_call = ToolCall(
        tool_name="from_scratch__read_file",
        arguments={"path": "data.txt"},
    )

    result = tool(tool_call=tool_call)

    assert result.error is False
    assert result.content == "safe content"


def test_default_tools_contains_read_file_tool() -> None:
    """Tests DEFAULT_TOOLS includes a ReadFileTool instance."""
    assert any(isinstance(t, ReadFileTool) for t in DEFAULT_TOOLS)
