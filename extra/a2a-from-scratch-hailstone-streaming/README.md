# A2A From-Scratch Hailstone (Streaming)

This framework's own `LLMAgent`, served over A2A (Agent2Agent protocol) via
`StreamingLLMAgentA2AExecutor` instead of the non-streaming
`LLMAgentA2AExecutor`. A sibling to `extra/a2a-from-scratch-hailstone`,
computing the same Hailstone sequence, but publishing an incremental
`TaskStatusUpdateEvent` after every step instead of a single terminal
update — the contrast `more-examples/ch10/streaming_llmagent_executor.ipynb`
demonstrates.

## Overview

The server exposes a single skill, `hailstone_sequence`, that computes the
full [Collatz conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture)
(Hailstone) sequence for a positive integer, one step at a time, until it
reaches 1:

- If `x` is even: next value is `x / 2`
- If `x` is odd: next value is `3x + 1`

`main.py` wraps an `LLMAgent` (equipped with a `next_number` tool) in
`StreamingLLMAgentA2AExecutor`, builds its `AgentCard` via
`build_streaming_agent_card()` (`capabilities.streaming=True`), and mounts
it on a FastAPI app — a genuinely separate OS process, not an in-process
background task, same as its non-streaming sibling.

## Installation

```bash
cd extra/a2a-from-scratch-hailstone-streaming
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
uv run uvicorn main:app --host 0.0.0.0 --port 9301
```

Configurable via environment variables:

| Variable          | Default                      | Purpose                                                |
|-------------------|-------------------------------|----------------------------------------------------------|
| `OLLAMA_MODEL`    | `qwen3:14b`                   | Model passed to `OllamaLLM`                                |
| `OLLAMA_HOST`     | unset (local Ollama)          | `OllamaLLM`'s `host` param, e.g. `https://ollama.com` for Ollama Cloud |
| `A2A_HOST`        | `0.0.0.0`                     | Host uvicorn binds to                                     |
| `A2A_PORT`        | `9301`                        | Port uvicorn binds to                                     |
| `A2A_URL`         | `http://localhost:{A2A_PORT}` | URL advertised in the agent card's `supported_interfaces` |

### Connect with a streaming client

Discover the card, then send a message with `ClientConfig(streaming=True)`
to receive incremental `TaskStatusUpdateEvent`s rather than a single
terminal response:

```python
from a2a.client import ClientConfig, create_client
from a2a.helpers import new_text_message
from a2a.types import Role as A2ARole, SendMessageRequest

client = await create_client(
    agent=agent_card,
    client_config=ClientConfig(streaming=True, httpx_client=httpx_client),
)
message = new_text_message(text="...", role=A2ARole.ROLE_USER)
async for chunk in client.send_message(SendMessageRequest(message=message)):
    ...  # one chunk per published event, not just the final one
```

## A2A Skill

| Name                 | Description                                            | Parameters                                    |
|----------------------|----------------------------------------------------------|--------------------------------------------------|
| `hailstone_sequence` | Computes the full Hailstone sequence for a positive integer, ending at 1 | task instruction naming the starting integer `x` |
