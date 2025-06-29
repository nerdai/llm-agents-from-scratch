"""Base Tool."""

from abc import ABC, abstractmethod
from typing import Any

from llm_agents_from_scratch.data_structures.tool import (
    ToolCall,
    ToolCallResult,
)


class BaseTool(ABC):
    """Base Tool Class."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this tool does."""

    @property
    @abstractmethod
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters."""

    @abstractmethod
    def __call__(
        self,
        tool_call: ToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ToolCallResult:
        """Execute the tool call.

        Args:
            tool_call (ToolCall): The tool call to execute.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.


        Returns:
            ToolCallResult: The result of the tool call execution.
        """


class AsyncBaseTool(ABC):
    """Async Base Tool Class."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this tool does."""

    @property
    @abstractmethod
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters."""

    @abstractmethod
    async def __call__(
        self,
        tool_call: ToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ToolCallResult:
        """Asynchronously execute the tool call.

        Args:
            tool_call (ToolCall): The tool call to execute.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.


        Returns:
            ToolCallResult: The result of the tool call execution.
        """
