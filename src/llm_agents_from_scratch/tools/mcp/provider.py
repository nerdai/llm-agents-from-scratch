"""MCP Tool Provider."""

import warnings
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from llm_agents_from_scratch.errors import (
    MCPWarning,
    MissingMCPServerParamsError,
)

if TYPE_CHECKING:
    from .tool import MCPTool


class MCPToolProvider:
    """MCP Tool Provider class."""

    def __init__(
        self,
        name: str,
        stdio_params: StdioServerParameters | None = None,
        streamable_http_url: str | None = None,
    ) -> None:
        """Initialize an MCPToolProvider.

        Args:
            name (str): A name identifier for this provider. Used to prefix
                tool names when creating MCPTool instances (e.g.,
                "{name}.{tool_name}").
            stdio_params (StdioServerParameters | None, optional): Parameters
                for connecting to an MCP server via stdio. If both this and
                `streamable_http_url` are provided, stdio will be used and
                HTTP will be ignored. Defaults to None.
            streamable_http_url (str | None, optional): URL for connecting to
                an MCP server via HTTP. Only used if `stdio_params` is None.
                Defaults to None.

        Raises:
            MissingMCPServerParamsError: If neither `stdio_params` nor
                `streamable_http_url` is provided.

        Warns:
            MCPWarning: Emitted if both `stdio_params` and
                `streamable_http_url` are provided (stdio will be prioritized).
        """
        if (stdio_params is None) and (streamable_http_url is None):
            msg = (
                "You must supply at least one of `stdio_params` or "
                "`streamable_http_url` to connect to an MCP server."
            )
            raise MissingMCPServerParamsError(msg)

        if stdio_params and streamable_http_url:
            msg = (
                "Both `stdio_params` and `streamable_http_url` were provided; "
                "`stdio_params` will be used and `streamable_http_url` ignored."
            )
            warnings.warn(msg, MCPWarning, stacklevel=2)

        self.name = name
        self.stdio_params = stdio_params
        self.streamable_http_url = streamable_http_url

    @asynccontextmanager
    async def session(self) -> AsyncIterator[ClientSession]:
        """An async context manager for creating a client session.

        Yields:
            ClientSession: An initialized MCP client session. Automatically
                closed when exiting the context.
        """
        if self.stdio_params:
            async with stdio_client(self.stdio_params) as (read, write):  # noqa: SIM117
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        else:
            async with streamablehttp_client(self.streamable_http_url) as (  # noqa: SIM117
                read_stream,
                write_stream,
                _,
            ):
                # Create a session using the client streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    yield session

    async def get_tools(self) -> list["MCPTool"]:
        """Fetch tools from the MCP server and create MCPTool instances."""
        from llm_agents_from_scratch.tools.mcp.tool import (  # noqa: PLC0415
            MCPTool,
        )

        async with self.session() as session:
            response = await session.list_tools()

        return [
            MCPTool(
                provider=self,
                name=f"{self.name}.{tool.name}",
                desc=tool.description,
                params_json_schema=tool.inputSchema,
                additional_annotations=tool.annotations,
            )
            for tool in response.tools
        ]
