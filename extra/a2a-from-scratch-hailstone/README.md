# A2A From-Scratch Hailstone

This framework's own `LLMAgent`, served over A2A (Agent2Agent protocol),
that computes the full Hailstone sequence for a positive integer. A
sibling to `extra/a2a-crewai-hailstone`, but built entirely with this
repo's own primitives instead of a different agent stack — proof that
`LLMAgentA2AExecutor` produces a genuine, independent A2A peer, not just
something this framework can talk to itself in-process.

## Overview

The server exposes a single skill, `hailstone_sequence`, that computes the
full [Collatz conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture)
(Hailstone) sequence for a positive integer, one step at a time, until it
reaches 1:

- If `x` is even: next value is `x / 2`
- If `x` is odd: next value is `3x + 1`

`main.py` wraps an `LLMAgent` (equipped with a `next_number` tool) in
`LLMAgentA2AExecutor`, builds its `AgentCard` via `build_agent_card()`,
and mounts it on a FastAPI app — the exact same construction pattern
demonstrated inline (without actually serving it) in Example 4a of
`examples/ch10.ipynb`. This app is what actually serves it, so Ch10's
loopback demo dispatches to a genuinely separate OS process, not an
in-process background task — a real A2A peer is remote, an
out-of-process black box, not a co-routine sharing the caller's
process.

## Installation

```bash
cd extra/a2a-from-scratch-hailstone
uv sync
```

`llm-agents-from-scratch` is installed from source (this repo, via
`[tool.uv.sources]` in `pyproject.toml`) rather than from PyPI: no
released version has A2A support yet (still under "Unreleased" in
`CHANGELOG.md` as of this writing). Once a release with it ships, this
should switch to a version constraint like the rest of this app's
dependencies.

Requires a running Ollama instance (`ollama serve`) with the configured
model pulled — defaults to `qwen3:14b` (`ollama pull qwen3:14b`).

## Usage

### Run the A2A server

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 9300
```

Configurable via environment variables:

| Variable          | Default                      | Purpose                                                |
|-------------------|-------------------------------|----------------------------------------------------------|
| `OLLAMA_MODEL`    | `qwen3:14b`                   | Model passed to `OllamaLLM`                                |
| `OLLAMA_HOST`     | unset (local Ollama)          | `OllamaLLM`'s `host` param, e.g. `https://ollama.com` for Ollama Cloud |
| `A2A_HOST`        | `0.0.0.0`                     | Host uvicorn binds to                                     |
| `A2A_PORT`        | `9300`                        | Port uvicorn binds to                                     |
| `A2A_URL`         | `http://localhost:{A2A_PORT}` | URL advertised in the agent card's `supported_interfaces` |

### Connect from an LLM Agent

Discover the card and register it as a peer via `A2AAgentSpec.from_url()`:

```python
from llm_agents_from_scratch.a2a import A2AAgentSpec
from llm_agents_from_scratch.agent import LLMAgentBuilder
from llm_agents_from_scratch.llms.ollama import OllamaLLM

spec = await A2AAgentSpec.from_url("http://localhost:9300")
agent = (
    await LLMAgentBuilder()
    .with_llm(OllamaLLM(model="qwen3:14b"))
    .with_a2a_agent(spec)
    .build()
)
```

## A2A Skill

| Name                 | Description                                            | Parameters                                    |
|----------------------|----------------------------------------------------------|--------------------------------------------------|
| `hailstone_sequence` | Computes the full Hailstone sequence for a positive integer, ending at 1 | task instruction naming the starting integer `x` |
