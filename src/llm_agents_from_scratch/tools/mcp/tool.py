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
        """Initialize an MCP Tool.

        Note:
            It is highly recommended to use `MCPToolProvider.get_tools()` to
            create MCPTool instances. It automatically names tools as
            "mcp__{provider_name}__{server_tool_name}" to avoid collisions
            across providers. When the tool is invoked, the provider prefix
            is stripped to call the tool by its original server-side name.

        Args:
            provider (MCPToolProvider): The provider that owns this tool and
                manages the connection to the MCP server.
            name (str): The fully qualified tool name. When created via
                `provider.get_tools()`, this follows the format
                "mcp__{provider_name}__{server_tool_name}".
            desc (str): A description of what the tool does.
            params_json_schema (dict[str, Any]): JSON schema defining the
                tool's input parameters.
            additional_annotations (ToolAnnotations | None, optional):
                Additional MCP tool annotations (hints for clients).
                Defaults to None.
        """
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
        session = await self.provider.session()
        # call tool
        result = await session.call_tool(
            name=self.name.removeprefix(f"mcp__{self.provider.name}__"),
            arguments=tool_call.arguments,
        )

        return ToolCallResult(
            tool_call_id=tool_call.id_,
            content=[el.model_dump() for el in result.content],
            error=result.isError,
        )
