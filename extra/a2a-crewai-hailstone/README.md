# A2A CrewAI Hailstone

A CrewAI agent, served over A2A (Agent2Agent protocol), that computes the
full Hailstone sequence for a positive integer. Built with a different
agent stack (CrewAI) than the rest of this repo on purpose — it's a genuine
external peer for Chapter 10's A2A examples, not another `LLMAgent`.

## Overview

The server exposes a single skill, `hailstone_sequence`, that computes the
full [Collatz conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture)
(Hailstone) sequence for a positive integer, one step at a time, until it
reaches 1:

- If `x` is even: next value is `x / 2`
- If `x` is odd: next value is `3x + 1`

Like `extra/mcp-hailstone`, the underlying `hailstone_step` primitive only
computes a single step. The difference here is who does the looping: a
CrewAI `Agent` (backed by an Ollama model via `crewai.LLM`) receives the
caller's task instruction, extracts the starting integer, then calls
`hailstone_step` repeatedly — feeding each result back in as the next
call's input — until the sequence reaches 1, the same way the book's
`LLMAgent` drives repeated tool calls itself rather than looping in the
tool.

## Installation

```bash
cd extra/a2a-crewai-hailstone
uv sync
```

Requires a running Ollama instance (`ollama serve`) with the configured
model pulled — defaults to `qwen3:14b`.

## Usage

### Run the A2A server

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 9200
```

Configurable via environment variables:

| Variable          | Default                      | Purpose                                                |
|-------------------|-------------------------------|----------------------------------------------------------|
| `OLLAMA_MODEL`    | `ollama/qwen3:14b`            | Model passed to `crewai.LLM`                              |
| `OLLAMA_BASE_URL` | `http://localhost:11434`      | Ollama server URL                                         |
| `A2A_HOST`        | `0.0.0.0`                     | Host uvicorn binds to                                     |
| `A2A_PORT`        | `9200`                        | Port uvicorn binds to                                     |
| `A2A_URL`         | `http://localhost:{A2A_PORT}` | URL advertised in the agent card's `supported_interfaces` |

### Connect from an LLM Agent

Discover the card and register it as a peer via `A2AAgentSpec.from_url()`:

```python
from llm_agents_from_scratch.a2a import A2AAgentSpec
from llm_agents_from_scratch.agent import LLMAgentBuilder
from llm_agents_from_scratch.llms.ollama import OllamaLLM

spec = await A2AAgentSpec.from_url("http://localhost:9200")
agent = (
    await LLMAgentBuilder()
    .with_llm(OllamaLLM(model="qwen3:14b"))
    .with_a2a_agent(spec)
    .build()
)
```

## Skill

| Name                 | Description                                            | Parameters                                    |
|----------------------|----------------------------------------------------------|--------------------------------------------------|
| `hailstone_sequence` | Computes the full Hailstone sequence for a positive integer, ending at 1 | task instruction naming the starting integer `x` |
