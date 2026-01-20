"""Unit tests for MCPToolProvider."""

from mcp import StdioServerParameters

from llm_agents_from_scratch.tools.mcp import MCPToolProvider


def test_mcp_tool_provider_init() -> None:
    """Tests initialization of an MCPToolProvider."""
    streamable_http_provider = MCPToolProvider(
        name="mock provider",
        streamable_http_url="https://mock-server-url.io",
    )

    assert streamable_http_provider.name == "mock provider"
    assert streamable_http_provider.stdio_params is None
    assert (
        streamable_http_provider.streamable_http_url
        == "https://mock-server-url.io"
    )

    stdio_params = StdioServerParameters(
        command="uv run",
        args=["fake.py"],
    )
    stdio_provider = MCPToolProvider(
        name="mock provider",
        stdio_params=stdio_params,
    )
    assert stdio_provider.name == "mock provider"
    assert stdio_provider.streamable_http_url is None
    assert stdio_provider.stdio_params == stdio_params
