"""MCP Tool."""

from typing import Any

from mcp.types import ToolAnnotations

from llm_agents_from_scratch.base.tool import AsyncBaseTool
from llm_agents_from_scratch.data_structures import ToolCall, ToolCallResult
from llm_agents_from_scratch.tools.mcp.provider import MCPToolProvider


class MCPTool(AsyncBaseTool):
    """MCP Tool Class."""

    def __init__(
        self,
        provider: MCPToolProvider,
        name: str,
        desc: str,
        params_json_schema: dict[str, Any],
        additional_annotations: ToolAnnotations | None = None,
    ) -> None:
        """Initialize an MCP Tool."""
        self.provider = provider
        self._name = name
        self._desc = desc
        self._params_json_schema = params_json_schema
        self.additional_annotations = additional_annotations

    @property
    def name(self) -> str:
        """Implements name property."""
        return self._name

    @property
    def description(self) -> str:
        """Implements description property."""
        return self._desc

    @property
    def parameters_json_schema(self) -> dict[str, Any]:
        """JSON schema for tool parameters."""
        return self._params_json_schema

    async def __call__(
        self,
        tool_call: ToolCall,
        *args: Any,
        **kwargs: Any,
    ) -> ToolCallResult:
        """Asynchronously execute the MCP tool call.

        Args:
            tool_call (ToolCall): The tool call to execute.
            *args (Any): Additional positional arguments forwarded to the tool.
            **kwargs (Any): Additional keyword arguments forwarded to the tool.

        Returns:
            ToolCallResult: The tool call result.
        """
        raise NotImplementedError
