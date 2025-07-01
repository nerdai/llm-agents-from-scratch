import re
from typing import Callable

import pytest
from pydantic import BaseModel, Field

from llm_agents_from_scratch.data_structures import ToolCall
from llm_agents_from_scratch.tools.pydantic_function import (
    PydanticFunction,
    PydanticFunctionTool,
    _validate_pydantic_function,
)


class ParamSet1(BaseModel):
    param1: int = Field(description="A mock param 1")
    param2: str = Field(description="A mock param 2", default="x")


def my_mock_fn_1(params: ParamSet1) -> str:
    return f"{params.param1} and {params.param2}"


def test__validate_pydantic_function() -> None:
    param_mdl = _validate_pydantic_function(my_mock_fn_1)

    assert param_mdl == ParamSet1


def invalid_pydantic_fn_1(foo: str) -> bool:
    return False


def invalid_pydantic_fn_2(params: str) -> bool:
    return False


def invalid_pydantic_fn_3(params) -> bool:
    return False


@pytest.mark.parametrize(
    ("func", "msg"),
    [
        (
            invalid_pydantic_fn_1,
            "Validation of `func` failed: Missing `params` argument.",
        ),
        (
            invalid_pydantic_fn_2,
            (
                "Validation of `func` failed: <class 'str'> is not"
                " a subclass of `~pydantic.BaseModel`."
            ),
        ),
        (
            invalid_pydantic_fn_3,
            (
                "Validation of `func` failed: `params` argument must have "
                "type annotation."
            ),
        ),
    ],
)
def test_validate_pydantic_function_raises_error(
    func: Callable,
    msg: str,
) -> None:
    """Tests all the cases where validation of a PydanticFunction fails."""
    with pytest.raises(RuntimeError, match=re.escape(msg)):
        _validate_pydantic_function(func)


def test_pydantic_function_protocol() -> None:
    """Test interface for PydanticFunction protocol."""
    assert callable(PydanticFunction)
    assert hasattr(PydanticFunction, "__call__")  # noqa: B004
    assert hasattr(PydanticFunction, "__name__")
    assert hasattr(PydanticFunction, "__doc__")


def test_init_pydantic_function_tool() -> None:
    """Tests PydanticFunctionTool initialization."""
    tool = PydanticFunctionTool(my_mock_fn_1)

    assert tool.name == "my_mock_fn_1"
    assert (
        tool.description == f"Tool for {my_mock_fn_1.__name__}"
    )  # default when None
    assert tool.parameters_json_schema == ParamSet1.model_json_schema()
    assert tool.func == my_mock_fn_1


def test_function_tool_call() -> None:
    """Tests a function tool call."""
    tool = PydanticFunctionTool(my_mock_fn_1, desc="mock desc")
    tool_call = ToolCall(
        tool_name="my_mock_fn_1",
        arguments={"param1": 1, "param2": "y"},
    )

    result = tool(tool_call=tool_call)

    assert result.content == "1 and y"
    assert result.error is False


def test_function_tool_call_returns_error() -> None:
    """Tests a function tool call raises error at validation of params."""
    tool = PydanticFunctionTool(my_mock_fn_1, desc="mock desc")
    tool_call = ToolCall(
        tool_name="my_mock_fn_1",
        arguments={"param1": "1.35", "param2": "y"},
    )

    result = tool(tool_call=tool_call)

    assert (
        "Input should be a valid integer, unable to parse string as an integer"
        in result.content
    )
    assert result.error is True
