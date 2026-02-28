# LLM Agents From Scratch

[![Linting](https://github.com/nerdai/llm-agents-from-scratch/actions/workflows/lint.yml/badge.svg)](https://github.com/nerdai/llm-agents-from-scratch/actions/workflows/lint.yml)
[![Unit Testing and Upload Coverage](https://github.com/nerdai/llm-agents-from-scratch/actions/workflows/unit_test.yml/badge.svg)](https://github.com/nerdai/llm-agents-from-scratch/actions/workflows/unit_test.yml)
[![codecov](https://codecov.io/gh/nerdai/llm-agents-from-scratch/graph/badge.svg?token=I1CXFJXEXK)](https://codecov.io/gh/nerdai/llm-agents-from-scratch)
[![DOI](https://zenodo.org/badge/1002134625.svg)](https://doi.org/10.5281/zenodo.15857308)
[![GitHub License](https://img.shields.io/github/license/nerdai/llm-agents-from-scratch)](https://github.com/nerdai/llm-agents-from-scratch/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/nerdai/llm-agents-from-scratch)

The companion library for *Build a Multi-Agent System — With MCP and A2A*
(Manning). Learn how LLM agents work by building one yourself, from first
principles, step by step.

> **Available now through Manning's Early Access Program (MEAP)** — buy today and get each chapter as it's completed.
> [Buy the Book →](https://hubs.la/Q03Q0h770)

---

## About

Multi-agent systems and the LLM agents that power them are among the most
discussed topics in AI today. There are already many capable frameworks out
there — the goal of this book isn't to replace them, but to help you deeply
understand how they work by having you build one yourself, from scratch.

All the code lives in the book's own hand-rolled agent framework, primarily
designed for educational purposes rather than production deployment. It will
give you the foundation to work more confidently with any other LLM agent
framework of your choosing, or even to build your own specialised solutions.

---

## From the Book

Each chapter builds on the last, progressively deepening your understanding from
core concepts to full multi-agent systems.

### Part 1 — Build Your First LLM Agent

| Ch | Title | Notebook |
| -- | ----- | -------- |
| 1 | What Are LLM Agents and Multi-Agent Systems? | — |
| 2 | Working with Tools | [Ch 2](https://masfromscratch.com/notebooks/ch02/) |
| 3 | Working with LLMs | [Ch 3](https://masfromscratch.com/notebooks/ch03/) |
| 4 | The LLM Agent Class | [Ch 4](https://masfromscratch.com/notebooks/ch04/) |

### Part 2 — Enhance Your LLM Agent

| Ch | Title | Notebook |
| -- | ----- | -------- |
| 5 | MCP Tools | [Ch 5](https://masfromscratch.com/notebooks/ch05/) |
| 6 | Skills | — |
| 7 | Memory | — |
| 8 | Human in the Loop | — |

### Part 3 — Building Multi-Agent Systems

| Ch | Title | Notebook |
| -- | ----- | -------- |
| 9 | Multi-Agent Systems with Agent2Agent | — |

---

## Capstone Projects

Capstones are larger, end-to-end projects that pull together what you have built
in the book and apply it to something closer to a real-world system.

| Capstone | Description | Notebook |
| -------- | ----------- | -------- |
| Monte Carlo Estimation of Pi | Orchestrate parallel tool calls to estimate π using the Monte Carlo method. | [Open](https://masfromscratch.com/capstones/one/capstone_1_ch05/) |
| Deep Research Agent | *Coming soon.* | — |
| OpenClaw Personal Assistant | *Coming soon.* | — |

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally (used as the default LLM backend)
- [uv](https://docs.astral.sh/uv/) for dependency management

### Installation

Clone the repository:

```sh
# SSH
git clone git@github.com:nerdai/llm-agents-from-scratch.git

# HTTPS
git clone https://github.com/nerdai/llm-agents-from-scratch.git

cd llm-agents-from-scratch
```

Install dependencies:

```sh
uv sync --all-extras --dev
```

### Quick Start

```python
from llm_agents_from_scratch.llms import OllamaLLM
from llm_agents_from_scratch.agent import LLMAgentBuilder
from llm_agents_from_scratch.tools import SimpleFunctionTool

def add(a: int, b: int) -> int:
    return a + b

llm = OllamaLLM(model="llama3.2")
tool = SimpleFunctionTool(fn=add)

agent = (
    LLMAgentBuilder()
    .with_llm(llm)
    .with_tools([tool])
    .build()
)

result = await agent.run("What is 3 + 5?")
print(result)
```

---

## Development

```sh
# Run all tests
make test

# Lint and format
make lint
make format

# Coverage report
make coverage-report
```

See [CLAUDE.md](CLAUDE.md) for full development guidance.

---

## Contributing

Bug reports, feature requests, and community project submissions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

- **Found a bug?** [Open an issue](https://github.com/nerdai/llm-agents-from-scratch/issues)
- **Built something cool?** [Share it on GitHub Discussions](https://github.com/nerdai/llm-agents-from-scratch/discussions)

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
