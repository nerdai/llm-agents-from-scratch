from typing import Any, Sequence

import pytest

from llm_agents_from_scratch.tools.function import (
    FunctionTool,
    function_signature_to_json_schema,
)


def my_mock_fn_1(
    param1: int,
    param2: str = "x",
    *args: Any,
    **kwargs: Any,
) -> bool:
    return False


def my_mock_fn_2(
    param1: int | None,
    param2: Sequence[int],
) -> str:
    return ""


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
    ],
)
def test_function_as_json_schema(func, properties, required) -> None:
    schema = function_signature_to_json_schema(func)

    assert schema["type"] == "object"
    assert schema["properties"] == properties
    assert schema["required"] == required


def test_function_tool_init() -> None:
    """Tests FunctionTool initialization."""
    tool = FunctionTool(my_mock_fn_1, desc="mock desc")

    assert tool.name == "my_mock_fn_1"
    assert tool.description == "mock desc"
    assert tool.parameters_json_schema == function_signature_to_json_schema(
        my_mock_fn_1,
    )
    assert tool.func == my_mock_fn_1
