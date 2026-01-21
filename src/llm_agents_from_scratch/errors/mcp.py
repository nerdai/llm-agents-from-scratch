"""Errors for MCP Tools."""

from .core import LLMAgentsFromScratchError, LLMAgentsFromScratchWarning


class MCPError(LLMAgentsFromScratchError):
    """Base error for all MCP-related exceptions."""

    pass


class MCPWarning(LLMAgentsFromScratchWarning):
    """Base warning for all MCP-related warnings."""

    pass


class MissingMCPServerParamsError(MCPError):
    """Raised when constructing an MCPToolProvider without MCP server params."""

    pass
