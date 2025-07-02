"""Pydantic Function Tool."""

import inspect
from typing import Any, Awaitable, Callable, Protocol, get_type_hints

from pydantic import BaseModel

from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import ToolCall, ToolCallResult


class PydanticFunction(Protocol):
    """PydanticFunction Protocol."""

    __name__: str
    __doc__: str | None

    def __call__(self, params: BaseModel, *args: Any, **kwargs: Any) -> Any:
        """Callable interface.

        Args:
            params (BaseModel): The function's params as a ~pydantic.BaseModel.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Any: The result of the function call.
        """
        ...  # pragma: no cover


class AsyncPydanticFunction(Protocol):
    """Asynchronous PydanticFunction Protocol."""

    __name__: str
    __doc__: str | None

    async def __call__(
        self,
        params: BaseModel,
        *args: Any,
        **kwargs: Any,
    ) -> Awaitable[Any]:
        """Asynchronous callable interface.

        Args:
            params (BaseModel): The function's params as a ~pydantic.BaseModel.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Awaitable[Any]: The result of the function call.
        """
        ...  # pragma: no cover


def _validate_pydantic_function(func: Callable) -> type[BaseModel]:
    """Validates func as a proper `PydanticFunction`.

    Args:
        func (Callable): The function to validate as `PydanticFunction`.

    Raises:
        RuntimeError: If validation of `func` fails.
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)  # resolves forward references

    if "params" not in sig.parameters:
        raise RuntimeError(
            "Validation of `func` failed: Missing `params` argument.",
        )

    if annotation := type_hints.get("params"):
        if not issubclass(annotation, BaseModel):
            msg = (
                f"Validation of `func` failed: {annotation} is not"
                " a subclass of `~pydantic.BaseModel`."
            )
            raise RuntimeError(msg)
    else:
        msg = (
            "Validation of `func` failed: `params` argument must have "
            "type annotation."
        )
        raise RuntimeError(msg)

    return annotation  # type: ignore [no-any-return]


class PydanticFunctionTool(BaseTool):
    """Pydantic function calling tool.

    Turn a Python function that takes in a ~pydantic.BaseModel params object
    into a tool for an LLM.

    Attributes:
        func: PydanticFunction to represent as a tool.
        params_mdl: The params BaseModel.
        desc: Description of the PydanticFunction.
    """

    def __init__(
        self,
        func: PydanticFunction,
        desc: str | None = None,
    ):
        """Initialize a PydanticFunctionTool.

        Args:
            func (PydanticFunction): The Pydantic function to expose as a tool
                to the LLM.
            desc (str | None, optional): Description of the function.
                Defaults to None.
        """
        self.func = func
        self.desc = desc or func.__doc__ or f"Tool for {func.__name__}"
        self.params_mdl = _validate_pydantic_function(func)

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
        return self.params_mdl.model_json_schema()

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
            params = self.params_mdl.model_validate(tool_call.arguments)
            # execute the function
            res = self.func(params)
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


class AsyncPydanticFunctionTool(BaseTool):
    """Async Pydantic function calling tool.

    Turn an async Python function that takes in a ~pydantic.BaseModel params
    object into a tool for an LLM.

    Attributes:
        func: AsyncPydanticFunction to represent as a tool.
        params_mdl: The params BaseModel.
        desc: Description of the PydanticFunction.
    """

    def __init__(
        self,
        func: AsyncPydanticFunction,
        desc: str | None = None,
    ):
        """Initialize an AsyncPydanticFunctionTool.

        Args:
            func (AsyncPydanticFunction): The async Pydantic function to expose
                as a tool to the LLM.
            desc (str | None, optional): Description of the function.
                Defaults to None.
        """
        self.func = func
        self.desc = desc or func.__doc__ or f"Tool for {func.__name__}"
        self.params_mdl = _validate_pydantic_function(func)

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
        return self.params_mdl.model_json_schema()

    async def __call__(
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
            params = self.params_mdl.model_validate(tool_call.arguments)
            # execute the function
            res = await self.func(params)
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
