# MCP Hailstone

A simple MCP (Model Context Protocol) server that exposes the Hailstone tool
built in earlier chapters as an MCP tool.

## Overview

This server exposes a single tool `hailstone_step_fn` that performs one step of
the [Collatz conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture)
(Hailstone sequence):

- If `x` is even: return `x / 2`
- If `x` is odd: return `3x + 1`

## Installation

```bash
cd extra/mcp-hailstone
uv sync
```

## Usage

### Run the MCP Inspector (development)

```bash
uv run mcp dev main.py
```

### Run as stdio server

```bash
uv run --with mcp mcp run main.py
```

### Connect from an LLM Agent

Use `MCPToolProvider` with `StdioServerParameters`:

```python
from mcp import StdioServerParameters

from llm_agents_from_scratch.agent import LLMAgentBuilder
from llm_agents_from_scratch.tools.mcp import MCPToolProvider

provider = MCPToolProvider(
    name="hailstone",
    stdio_params=StdioServerParameters(
        command="uv",
        args=["run", "--with", "mcp", "mcp", "run", "main.py"],
        cwd="extra/mcp-hailstone",
    ),
)
agent = await LLMAgentBuilder().with_mcp_provider(provider).build()
```

## Tool

| Name                | Description                                      | Parameters |
|---------------------|--------------------------------------------------|------------|
| `hailstone_step_fn` | Performs a single step of the Hailstone sequence | `x: int`   |
