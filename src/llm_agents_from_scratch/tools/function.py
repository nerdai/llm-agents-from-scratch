"""Function Tool."""

import inspect
from typing import Any, Callable, get_type_hints

from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import ToolCall, ToolCallResult


def function_signature_to_json_schema(func: Callable) -> dict[str, Any]:
    """Turn a function signature into a JSON schema.

    Inspects the signature of the function and maps types to the appropriate
    JSON schema type.

    Args:
        func (Callable): The function for which to get the JSON schema.

    Returns:
        dict[str, Any]: The JSON schema
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)  # resolves forward references
    python_to_json_schema_type = {
        str: "string",
        int: "number",
        float: "number",
        dict: "object",
        list: "array",
        type(None): "null",
        bool: "boolean",
    }

    properties = {}
    required = []
    for param in sig.parameters.values():
        # skip args and kwargs
        if param.name.startswith(("args", "kwargs")):
            continue

        # get type annotations
        annotation = type_hints.get(param.name, param.annotation)

        # get json schema
        if annotation in python_to_json_schema_type:
            this_params_json_schema = {
                "type": python_to_json_schema_type[annotation],
            }
        else:
            # fallback schema, that accepts everything
            this_params_json_schema = {}
        properties[param.name] = this_params_json_schema

        # check if param is required
        if param.default == inspect._empty:
            required.append(param.name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


class FunctionTool(BaseTool):
    """Function calling tool.

    Turn a Python function into a tool for an LLM.
    """

    def __init__(self, func: Callable, desc: str) -> None:
        """Initialize a FunctionTool.

        Args:
            func (Callable): The Python function to expose as a tool to the LLM.
            desc (str): Description of the function.
        """
        self.func = func
        self.desc = desc

    @property
    def name(self) -> str:
        """Name of function tool."""
        return self.func.__name__

    @property
    def description(self) -> str:
        """Description of what this function tool does."""
        return self.desc

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters."""
        return function_signature_to_json_schema(self.func)

    def __call__(
        self,
        tool_call: ToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ToolCallResult:
        """Execute the function tool with a Toolcall."""
        raise NotImplementedError
