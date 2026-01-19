"""MCP Tool Provier."""

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from llm_agents_from_scratch.errors import MissingMCPServerParamsError

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
            name (str): _description_
            stdio_params (StdioServerParameters | None, optional):
                _description_. Defaults to None.
            streamable_http_url (str | None, optional):
                _description_. Defaults to None.
        """
        if (stdio_params is None) and (streamable_http_url is None):
            msg = (
                "You must supply at least one of `stdio_params` or "
                "`streamable_http_url` to connect to an MCP server."
            )
            raise MissingMCPServerParamsError(msg)

        self.name = str
        self.stdio_params = StdioServerParameters
        self.streamable_http_url = streamable_http_url

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[ClientSession]:
        """An async context manager for creatting a client session.

        Yields:
            Generator[ClientSession]: _description_
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
        """List tools."""
        raise NotImplementedError
