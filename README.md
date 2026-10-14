# LLM Agents From Scratch

[![Linting](https://github.com/nerdai/llm-agents-from-scratch/actions/workflows/lint.yml/badge.svg)](https://github.com/nerdai/llm-agents-from-scratch/actions/workflows/lint.yml)
[![Unit Testing and Upload Coverage](https://github.com/nerdai/llm-agents-from-scratch/actions/workflows/unit_test.yml/badge.svg)](https://github.com/nerdai/llm-agents-from-scratch/actions/workflows/unit_test.yml)
[![codecov](https://codecov.io/gh/nerdai/llm-agents-from-scratch/graph/badge.svg?token=I1CXFJXEXK)](https://codecov.io/gh/nerdai/llm-agents-from-scratch)
[![DOI](https://zenodo.org/badge/1002134625.svg)](https://doi.org/10.5281/zenodo.15857308)
[![GitHub License](https://img.shields.io/github/license/nerdai/llm-agents-from-scratch)](https://github.com/nerdai/llm-agents-from-scratch/blob/main/LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/nerdai/llm-agents-from-scratch)
<a href="https://lightning.ai/nerdai/templates/build-a-multi-agent-system-from-scratch"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open In Studio" height="20"/></a>

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
| 6 | Skills | [Ch 6](https://masfromscratch.com/notebooks/ch06/) |
| 7 | Memory | [Ch 7](https://masfromscratch.com/notebooks/ch07/) |
| 8 | Human in the Loop | [Ch 8](https://masfromscratch.com/notebooks/ch08/) |

### Part 3 — Building Multi-Agent Systems

| Ch | Title | Notebook |
| -- | ----- | -------- |
| 9 | Assembling MAS with Subagents | [Ch 9](https://masfromscratch.com/notebooks/ch09/) |
| 10 | Assembling Interoperable MAS with A2A | [Ch 10](https://masfromscratch.com/notebooks/ch10/) |

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

## Running the Notebooks

I recommend running the notebooks on **Ollama Cloud**, which serves larger
open-source models on Ollama's own hardware. These models are much more
capable than what an everyday laptop can run, and Ollama keeps rotating in
the newest, best-performing ones.

### Option A — Ollama Cloud (recommended)

1. Create an API key at <https://ollama.com/settings/keys>.
2. Export it: `export OLLAMA_API_KEY=<your key>`.

Then launch Jupyter from the project root:

```sh
uv run --with jupyter jupyter lab
```

The notebooks detect the key and use the default cloud model. I pick that
model by benchmarking the available Ollama Cloud models on how well each one
runs the notebooks end to end, and an automated run checks every notebook
against it weekly (see the
[notebook status page](https://masfromscratch.com/notebook-status/)). To try
another cloud model from <https://ollama.com/search?c=cloud>, set the
`OLLAMA_MODEL` environment variable to its name.

### Option B — Local Ollama

Install [Ollama](https://ollama.com/download) and pull the model:

```sh
ollama pull qwen3:14b
```

With no `OLLAMA_API_KEY` set, the notebooks start the Ollama service for you,
so there's no need to run `ollama serve`. A 14B model needs about 16 GB of RAM
and a dedicated GPU. You can set `OLLAMA_MODEL` to a smaller model (such as
`qwen3:4b`), but smaller models may not follow tool-use instructions
reliably, especially in later chapters.

### Option C — Lightning AI (local models on a free GPU)

[Lightning AI](https://lightning.ai) offers about 22 free GPU compute hours a
month, enough for several sessions on an L4.

1. [Create a free account](https://lightning.ai) if you don't have one.
2. Open the [book template](https://lightning.ai/nerdai/templates/build-a-multi-agent-system-from-scratch).
3. Click **Clone** and make sure it's the latest version.
4. Launch the Studio with an **L4 GPU**.
5. Open any notebook. Ollama and the model are already set up.

### OpenAI or Anthropic

The notebooks also run on OpenAI and Anthropic models. Install the extra
(`uv sync --extra openai` or `uv sync --extra anthropic`), export
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, and set `LLM_PROVIDER=openai` (or
`anthropic`). The "Setting up the LLM" section at the top of each notebook
has the details.

---

## Getting Started

### Prerequisites

- Python 3.10+
- An [Ollama Cloud](https://ollama.com/settings/keys) API key (recommended) or a
  local [Ollama](https://ollama.com/download) install; see
  [Running the Notebooks](#running-the-notebooks)
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
