# A2A CrewAI Hailstone

A CrewAI agent, served over A2A (Agent2Agent protocol), that exposes the
Hailstone tool built in earlier chapters as an A2A skill. Built with a
different agent stack (CrewAI) than the rest of this repo on purpose — it's
a genuine external peer for Chapter 10's A2A examples, not another
`LLMAgent`.

## Overview

The server exposes a single skill, `hailstone_step`, that performs one step
of the [Collatz conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture)
(Hailstone sequence):

- If `x` is even: return `x / 2`
- If `x` is odd: return `3x + 1`

A CrewAI `Agent` (backed by an Ollama model via `crewai.LLM`) receives the
caller's task instruction, extracts the integer, and calls the
`hailstone_step` tool to compute the result — the LLM does the parsing, the
tool does the arithmetic.

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

| Name | Description | Parameters |
|---|---|---|
| `hailstone_step` | Performs a single step of the Hailstone sequence | task instruction naming the integer `x` |
