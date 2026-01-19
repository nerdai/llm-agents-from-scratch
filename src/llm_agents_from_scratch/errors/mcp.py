"""Errors for MCP Tools."""

from .core import LLMAgentsFromScratchError


class MCPError(LLMAgentsFromScratchError):
    """Base error for all MCP-related exceptions."""

    pass


class MissingMCPServerParamsError(MCPError):
    """Raised when constructing an MCPToolProvider without MCP server params."""

    pass
