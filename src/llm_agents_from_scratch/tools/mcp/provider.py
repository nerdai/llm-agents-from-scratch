"""MCP Tool Provier."""

from mcp import StdioServerParameters

from ...errors import MissingMCPServerParamsError


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
