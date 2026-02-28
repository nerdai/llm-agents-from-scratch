---
hide:
  - navigation
  - toc
---

<!-- markdownlint-disable-file MD041 MD033 MD042 -->

<div class="landing-hero" markdown>
<div class="landing-hero__inner" markdown>
<div class="landing-hero__text" markdown>

# Build a Multi-Agent System From Scratch

The companion library for *Build a Multi-Agent System — With MCP and A2A*
(Manning). Build intelligent LLM agents in Python, from abstract base classes
to full MCP-powered multi-agent systems.

[Get Started :octicons-arrow-right-24:](notebooks/ch02.ipynb){ .md-button .md-button--primary }
[GitHub :fontawesome-brands-github:](https://github.com/nerdai/llm-agents-from-scratch){ .md-button .landing-hero__btn-secondary }

</div>
<div class="landing-hero__book">
<img src="assets/no-bg-book-cover.svg" alt="Build a Multi-Agent System book cover">
</div>
</div>
</div>

## What's Inside

<div class="landing-grid" markdown>

<div class="landing-card" markdown>
### :material-code-braces: Abstract Base Classes

Clean, extensible abstractions for `LLM`, `BaseTool`, and `LLMAgent` built on
Python's ABC pattern — easy to understand, easy to extend.
</div>

<div class="landing-card" markdown>
### :material-connection: MCP Integration

Native [Model Context Protocol](https://modelcontextprotocol.io) support via
`MCPTool` and `MCPToolProvider` — connect any MCP server as an external tool
provider.
</div>

<div class="landing-card" markdown>
### :material-lightning-bolt: Async First

Full `async`/`await` support with asyncio for high-performance concurrent
agent execution. Both sync and async tool variants are included.
</div>

<div class="landing-card" markdown>
### :material-shield-check: Type Safe

Strict `mypy` type hints throughout, with [Pydantic](https://docs.pydantic.dev)
models for all data structures — correctness from definition to runtime.
</div>

<div class="landing-card" markdown>
### :material-swap-horizontal: Pluggable LLMs

Ships with `OllamaLLM` and `OpenAILLM`. Bring your own backend by extending
`BaseLLM` with four abstract methods.
</div>

<div class="landing-card" markdown>
### :material-notebook-outline: Chapter Notebooks

Step-by-step Jupyter notebooks accompany each chapter — run them locally or
follow along in the docs.
</div>

</div>

---

## From the Book

Each chapter builds on the last, taking you from first principles to
production-ready multi-agent systems:

| Chapter | Topic | Notebook |
| ------- | ----- | -------- |
| 2 | Core abstractions — `BaseLLM`, `BaseTool`, `LLMAgent` | [Ch 2](notebooks/ch02.ipynb) |
| 3 | Tool calling and structured outputs | [Ch 3](notebooks/ch03.ipynb) |
| 4 | Async agents and multi-step rollouts | [Ch 4](notebooks/ch04.ipynb) |
| 5 | MCP integration and external tool providers | [Ch 5](notebooks/ch05.ipynb) |
