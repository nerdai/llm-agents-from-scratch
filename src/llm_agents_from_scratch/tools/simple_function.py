"""Function Tool."""

import inspect
from typing import Any, Awaitable, Callable, get_type_hints

from jsonschema import validate

from llm_agents_from_scratch.base.tool import AsyncBaseTool, BaseTool
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
        tuple: "array",
        bytes: "string",
        set: "array",
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


class SimpleFunctionTool(BaseTool):
    """Simple function calling tool.

    Turn a Python function into a tool for an LLM. Uses manual validation for
    JSON schema.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        desc: str | None = None,
    ) -> None:
        """Initialize a SimpleFunctionTool.

        Args:
            func (Callable): The Python function to expose as a tool to the LLM.
            desc (str | None, optional): Description of the function.
                Defaults to None.
        """
        self.func = func
        self.desc = desc or func.__doc__ or f"Tool for {func.__name__}"

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
        """Execute the function tool with a ToolCall.

        Args:
            tool_call (ToolCall): The ToolCall to execute.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ToolCallResult: The result of the tool call execution.
        """
        try:
            # validate the arguments
            validate(tool_call.arguments, schema=self.parameters_json_schema)
            # execute the function
            res = self.func(**tool_call.arguments)
            content = str(res)
            error = False
        except Exception as e:
            content = f"Failed to execute function call: {e}"
            error = True

        return ToolCallResult(
            tool_call=tool_call,
            content=content,
            error=error,
        )


class AsyncSimpleFunctionTool(AsyncBaseTool):
    """Simple function calling tool.

    Turn a Python function into a tool for an LLM. Uses manual validation for
    JSON schema.
    """

    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        desc: str | None = None,
    ) -> None:
        """Initialize a SimpleFunctionTool.

        Args:
            func (Callable[..., Awaitable[Any]]): Async function.
            desc (str | None, optional): Description of the function.
                Defaults to None.
        """
        self.func = func
        self.desc = desc or func.__doc__ or f"Tool for {func.__name__}"

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

    async def __call__(
        self,
        tool_call: ToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ToolCallResult:
        """Asynchronously execute the function tool with a ToolCall.

        Args:
            tool_call (ToolCall): The ToolCall to execute.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            ToolCallResult: The result of the tool call execution.
        """
        try:
            # validate the arguments
            validate(tool_call.arguments, schema=self.parameters_json_schema)
            # execute the function
            res = await self.func(**tool_call.arguments)
            content = str(res)
            error = False
        except Exception as e:
            content = f"Failed to execute function call: {e}"
            error = True

        return ToolCallResult(
            tool_call=tool_call,
            content=content,
            error=error,
        )
