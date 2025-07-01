import asyncio
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

import pytest

from llm_agents_from_scratch.data_structures.tool import ToolCall
from llm_agents_from_scratch.tools.simple_function import (
    AsyncSimpleFunctionTool,
    SimpleFunctionTool,
    function_signature_to_json_schema,
)


def my_mock_fn_1(
    param1: int,
    param2: str = "x",
    *args: Any,
    **kwargs: Any,
) -> str:
    return f"{param1} and {param2}"


def my_mock_fn_2(
    param1: int | None,
    param2: Sequence[int],
) -> str:
    return ""


async def my_mock_fn_3(
    param1: int,
    param2: str = "x",
    *args: Any,
    **kwargs: Any,
) -> str:
    await asyncio.sleep(0.1)
    return f"{param1} and {param2}"


@pytest.mark.parametrize(
    ("func", "properties", "required"),
    [
        (
            my_mock_fn_1,
            {
                "param1": {
                    "type": "number",
                },
                "param2": {
                    "type": "string",
                },
            },
            ["param1"],
        ),
        (
            my_mock_fn_2,
            {
                "param1": {},
                "param2": {},
            },
            ["param1", "param2"],
        ),
        (
            my_mock_fn_3,
            {
                "param1": {
                    "type": "number",
                },
                "param2": {
                    "type": "string",
                },
            },
            ["param1"],
        ),
    ],
)
def test_function_as_json_schema(func, properties, required) -> None:
    schema = function_signature_to_json_schema(func)

    assert schema["type"] == "object"
    assert schema["properties"] == properties
    assert schema["required"] == required


def test_function_tool_init() -> None:
    """Tests SimpleFunctionTool initialization."""
    tool = SimpleFunctionTool(my_mock_fn_1, desc="mock desc")

    assert tool.name == "my_mock_fn_1"
    assert tool.description == "mock desc"
    assert tool.parameters_json_schema == function_signature_to_json_schema(
        my_mock_fn_1,
    )
    assert tool.func == my_mock_fn_1


@patch("llm_agents_from_scratch.tools.simple_function.validate")
def test_function_tool_call(mock_validate: MagicMock) -> None:
    """Tests a function tool call."""
    tool = SimpleFunctionTool(my_mock_fn_1, desc="mock desc")
    tool_call = ToolCall(
        tool_name="my_mock_fn_1",
        arguments={"param1": 1, "param2": "y"},
    )

    result = tool(tool_call=tool_call)

    assert result.content == "1 and y"
    mock_validate.assert_called_once_with(
        tool_call.arguments,
        schema=tool.parameters_json_schema,
    )
    assert result.error is False


def test_function_tool_call_returns_error() -> None:
    """Tests a function tool call raises error at validation of params."""
    tool = SimpleFunctionTool(my_mock_fn_1, desc="mock desc")
    tool_call = ToolCall(
        tool_name="my_mock_fn_1",
        arguments={"param1": "1", "param2": "y"},
    )

    result = tool(tool_call=tool_call)

    assert (
        "Failed to execute function call: '1' is not of type 'number'"
        in result.content
    )
    assert result.error is True


# async
def test_async_function_tool_init() -> None:
    """Tests AsyncSimpleFunctionTool initialization."""
    tool = AsyncSimpleFunctionTool(my_mock_fn_3, desc="mock desc")

    assert tool.name == "my_mock_fn_3"
    assert tool.description == "mock desc"
    assert tool.parameters_json_schema == function_signature_to_json_schema(
        my_mock_fn_3,
    )
    assert tool.func == my_mock_fn_3
    assert asyncio.iscoroutinefunction(tool.func)


@pytest.mark.asyncio
@patch("llm_agents_from_scratch.tools.simple_function.validate")
async def test_async_function_tool_call(mock_validate: MagicMock) -> None:
    """Tests a function tool call."""
    tool = AsyncSimpleFunctionTool(my_mock_fn_3, desc="mock desc")
    tool_call = ToolCall(
        tool_name="my_mock_fn_1",
        arguments={"param1": 1, "param2": "y"},
    )

    result = await tool(tool_call=tool_call)

    assert result.content == "1 and y"
    mock_validate.assert_called_once_with(
        tool_call.arguments,
        schema=tool.parameters_json_schema,
    )
    assert result.error is False


@pytest.mark.asyncio
async def test_async_function_tool_call_returns_error() -> None:
    """Tests a function tool call."""
    tool = AsyncSimpleFunctionTool(my_mock_fn_1, desc="mock desc")
    tool_call = ToolCall(
        tool_name="my_mock_fn_1",
        arguments={"param1": "1", "param2": "y"},
    )

    result = await tool(tool_call=tool_call)

    assert (
        "Failed to execute function call: '1' is not of type 'number'"
        in result.content
    )
    assert result.error is True
