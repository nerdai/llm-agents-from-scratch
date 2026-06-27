"""Unit tests for default tools."""

from pathlib import Path
from unittest.mock import patch

from llm_agents_from_scratch.data_structures import ToolCall
from llm_agents_from_scratch.tools.default import (
    DEFAULT_TOOLS,
    HumanInputTool,
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
    """Tests PythonInterpreterTool schema has path, code, stdin, cwd."""
    tool = PythonInterpreterTool()
    schema = tool.parameters_json_schema
    assert schema["type"] == "object"
    assert "path" in schema["properties"]
    assert "code" in schema["properties"]
    assert "stdin" in schema["properties"]
    assert "cwd" in schema["properties"]
    assert "required" not in schema


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


def test_python_interpreter_tool_error_on_missing_path_and_code() -> None:
    """Tests PythonInterpreterTool returns error when neither path nor code."""
    tool = PythonInterpreterTool()
    tool_call = ToolCall(
        tool_name="from_scratch__python_interpreter",
        arguments={},
    )

    result = tool(tool_call=tool_call)

    assert result.error is True
    assert "Missing" in result.content


def test_python_interpreter_tool_runs_inline_code() -> None:
    """Tests PythonInterpreterTool executes inline code via 'code' param."""
    tool = PythonInterpreterTool()
    tool_call = ToolCall(
        tool_name="from_scratch__python_interpreter",
        arguments={"code": 'print("hello from inline")'},
    )

    result = tool(tool_call=tool_call)

    assert result.error is False
    assert "hello from inline" in result.content


def test_python_interpreter_tool_inline_code_imports_from_cwd(
    tmp_path: Path,
) -> None:
    """Tests inline code can import modules from cwd."""
    (tmp_path / "mymod.py").write_text("VALUE = 42\n")
    tool = PythonInterpreterTool()
    tool_call = ToolCall(
        tool_name="from_scratch__python_interpreter",
        arguments={
            "code": "import mymod; print(mymod.VALUE)",
            "cwd": str(tmp_path),
        },
    )

    result = tool(tool_call=tool_call)

    assert result.error is False
    assert "42" in result.content


def test_python_interpreter_tool_inline_code_error() -> None:
    """Tests inline code that raises returns a RuntimeError result."""
    tool = PythonInterpreterTool()
    tool_call = ToolCall(
        tool_name="from_scratch__python_interpreter",
        arguments={"code": "raise ValueError('boom')"},
    )

    result = tool(tool_call=tool_call)

    assert result.error is True
    assert "RuntimeError" in result.content


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


def test_python_interpreter_tool_passes_stdin(tmp_path: Path) -> None:
    """Tests PythonInterpreterTool pipes stdin text into the script."""
    script = tmp_path / "echo_stdin.py"
    script.write_text("import sys\nprint(sys.stdin.read().strip())")
    tool = PythonInterpreterTool()
    tool_call = ToolCall(
        tool_name="from_scratch__python_interpreter",
        arguments={"path": str(script), "stdin": "hello stdin"},
    )

    result = tool(tool_call=tool_call)

    assert result.error is False
    assert "hello stdin" in result.content


def test_default_tools_contains_python_interpreter_tool() -> None:
    """Tests DEFAULT_TOOLS includes a PythonInterpreterTool instance."""
    assert any(isinstance(t, PythonInterpreterTool) for t in DEFAULT_TOOLS)


# --- HumanInputTool ---


def test_human_input_tool_name() -> None:
    """Tests HumanInputTool.name."""
    tool = HumanInputTool()
    assert tool.name == "human_input"


def test_human_input_tool_description() -> None:
    """Tests HumanInputTool.description mentions human and input."""
    tool = HumanInputTool()
    assert "human" in tool.description.lower()


def test_human_input_tool_parameters_json_schema() -> None:
    """Tests HumanInputTool schema has required prompt and optional choices."""
    tool = HumanInputTool()
    schema = tool.parameters_json_schema
    assert schema["type"] == "object"
    assert "prompt" in schema["properties"]
    assert "choices" in schema["properties"]
    assert schema["required"] == ["prompt"]


def test_human_input_tool_returns_response() -> None:
    """Tests HumanInputTool returns the human's response as content."""
    tool = HumanInputTool()
    tool_call = ToolCall(
        tool_name="human_input",
        arguments={"prompt": "What is your name?"},
    )

    with patch("rich.prompt.Prompt.ask", return_value="Alice"):
        result = tool(tool_call=tool_call)

    assert result.error is False
    assert result.content == "Alice"


def test_human_input_tool_passes_prompt_to_input() -> None:
    """Tests HumanInputTool renders the prompt in a Panel and uses > inline."""
    tool = HumanInputTool()
    tool_call = ToolCall(
        tool_name="human_input",
        arguments={"prompt": "How old are you?"},
    )

    with (
        patch("rich.console.Console.print"),
        patch("rich.prompt.Prompt.ask", return_value="30") as mock_ask,
    ):
        tool(tool_call=tool_call)

    mock_ask.assert_called_once_with(
        ">",
        console=mock_ask.call_args.kwargs["console"],
    )


def test_human_input_tool_missing_prompt_defaults_to_empty() -> None:
    """Tests HumanInputTool defaults to empty string when prompt is absent."""
    tool = HumanInputTool()
    tool_call = ToolCall(
        tool_name="human_input",
        arguments={},
    )

    with (
        patch("rich.console.Console.print"),
        patch("rich.prompt.Prompt.ask", return_value="ok") as mock_ask,
    ):
        tool(tool_call=tool_call)

    mock_ask.assert_called_once_with(
        ">",
        console=mock_ask.call_args.kwargs["console"],
    )


def test_human_input_tool_choices_passed_to_prompt() -> None:
    """Tests HumanInputTool passes choices to Prompt.ask when provided."""
    tool = HumanInputTool()
    tool_call = ToolCall(
        tool_name="human_input",
        arguments={"prompt": "Pick one:", "choices": ["yes", "no"]},
    )

    with (
        patch("rich.console.Console.print"),
        patch("rich.prompt.Prompt.ask", return_value="yes") as mock_ask,
    ):
        result = tool(tool_call=tool_call)

    mock_ask.assert_called_once_with(
        ">",
        choices=["yes", "no"],
        console=mock_ask.call_args.kwargs["console"],
    )
    assert result.error is False
    assert result.content == "yes"


def test_human_input_tool_eof_error() -> None:
    """Tests HumanInputTool returns error result on EOFError."""
    tool = HumanInputTool()
    tool_call = ToolCall(
        tool_name="human_input",
        arguments={"prompt": "Enter value:"},
    )

    with patch("rich.prompt.Prompt.ask", side_effect=EOFError):
        result = tool(tool_call=tool_call)

    assert result.error is True
    assert result.content is not None
    assert "stdin" in result.content.lower()


def test_human_input_tool_keyboard_interrupt() -> None:
    """Tests HumanInputTool returns error result on KeyboardInterrupt."""
    tool = HumanInputTool()
    tool_call = ToolCall(
        tool_name="human_input",
        arguments={"prompt": "Enter value:"},
    )

    with patch("rich.prompt.Prompt.ask", side_effect=KeyboardInterrupt):
        result = tool(tool_call=tool_call)

    assert result.error is True
    assert result.content is not None
    assert "declined" in result.content.lower()
