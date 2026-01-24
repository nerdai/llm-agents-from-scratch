"""Main application."""

from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Hailstone")


@mcp.tool()
async def hailstone_step_fn(x: int) -> int:
    """Performs a single step of the Hailstone sequence."""
    if x % 2 == 0:
        return x // 2
    return 3 * x + 1


if __name__ == "__main__":
    mcp.run(transport="stdio")  # pragma: no cover
