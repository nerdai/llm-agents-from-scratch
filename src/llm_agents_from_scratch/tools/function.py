"""Function Tool."""

from typing import Any, Callable

from llm_agents_from_scratch.base.tool import BaseTool
from llm_agents_from_scratch.data_structures import ToolCall, ToolCallResult


def to_camel_case(snake_str: str) -> str:
    """Convert a string to camel case.

    Args:
        snake_str (str): The string to convert to camel case.
    """
    elements = snake_str.split("_")
    return "".join(w.capitalize() for w in elements)


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
        return to_camel_case(self.func.__name__)

    @property
    def description(self) -> str:
        """Description of what this function tool does."""
        return self.desc

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters."""
        return {}

    def __call__(
        self,
        tool_call: ToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ToolCallResult:
        """Execute the function tool with a Toolcall."""
        raise NotImplementedError
