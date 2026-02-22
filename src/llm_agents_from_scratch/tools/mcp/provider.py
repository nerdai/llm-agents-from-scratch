"""MCP Tool Provider."""

import asyncio
import warnings
from typing import TYPE_CHECKING

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
        streamable_http_headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize an MCPToolProvider.

        Args:
            name (str): A name identifier for this provider. Used to prefix
                tool names when creating MCPTool instances (e.g.,
                "mcp__{name}__{tool_name}").
            stdio_params (StdioServerParameters | None, optional): Parameters
                for connecting to an MCP server via stdio. If both this and
                `streamable_http_url` are provided, stdio will be used and
                HTTP will be ignored. Defaults to None.
            streamable_http_url (str | None, optional): URL for connecting to
                an MCP server via HTTP. Only used if `stdio_params` is None.
                Defaults to None.
            streamable_http_headers (dict[str, str] | None, optional): HTTP
                headers to include with every request to the MCP server (e.g.,
                ``{"Authorization": "Bearer <token>"}``). Only used when
                connecting via HTTP. Defaults to None.

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
        self.streamable_http_headers = streamable_http_headers
        # initialize session management attributes
        self._shutdown_event = asyncio.Event()
        self._session_ready = asyncio.Event()
        self._session_task: asyncio.Task | None = None
        self._session: ClientSession | None = None

    async def _create_session(self) -> None:
        """Create and maintain persistent session until shutdown signal.

        This method runs as a background task, keeping the MCP session
        alive by holding the context managers open. When the shutdown
        event is set, the context managers exit gracefully.
        """
        if self.stdio_params:
            async with stdio_client(self.stdio_params) as (read, write):  # noqa: SIM117
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._session_ready.set()

                    # Wait for shutdown signal
                    await self._shutdown_event.wait()
        else:
            async with streamablehttp_client(  # noqa: SIM117
                self.streamable_http_url,
                self.streamable_http_headers,
            ) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._session = session
                    self._session_ready.set()

                    # Wait for shutdown signal
                    await self._shutdown_event.wait()

    async def session(self) -> ClientSession:
        """Get the persistent session.

        Returns:
            ClientSession: An initialized MCP client session.

        Note:
            This method uses lazy initialization - the session is created
            on the first call and reused for subsequent calls.
        """
        if not self._session_ready.is_set():
            self._session_task = asyncio.create_task(self._create_session())
            await self._session_ready.wait()
        return self._session  # type: ignore[return-value]

    async def get_tools(self) -> list["MCPTool"]:
        """Fetch tools from the MCP server and create MCPTool instances.

        Returns:
            list[MCPTool]: A list of MCPTool instances representing the
                tools available from the MCP server.
        """
        from llm_agents_from_scratch.tools.mcp.tool import (  # noqa: PLC0415
            MCPTool,
        )

        session = await self.session()
        response = await session.list_tools()

        return [
            MCPTool(
                provider=self,
                name=f"mcp__{self.name}__{tool.name}",
                desc=tool.description,
                params_json_schema=tool.inputSchema,
                additional_annotations=tool.annotations,
            )
            for tool in response.tools
        ]

    async def close(self) -> None:
        """Close the persistent session and clean up resources.

        Note:
            For short-lived scripts, calling close() is optional as the OS
            will clean up subprocess resources when your program exits.
            For long-running applications, you should call close() to
            prevent resource leaks.
        """
        if self._session_task:
            self._shutdown_event.set()
            try:
                await self._session_task
            finally:
                self._session = None
                self._session_task = None
                self._shutdown_event.clear()
                self._session_ready.clear()
