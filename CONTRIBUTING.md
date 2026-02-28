# Contributing

Thank you for your interest in contributing to **LLM Agents From Scratch**!
This project is the companion library for *Build a Multi-Agent System — With
MCP and A2A* (Manning). Contributions from readers and the wider community help
make it better for everyone.

---

## Reporting Bugs

If something isn't working as expected — a broken notebook, an incorrect
result, a library error — please open an issue:

1. Check the [existing issues](https://github.com/nerdai/llm-agents-from-scratch/issues)
   first to avoid duplicates.
2. [Open a new issue](https://github.com/nerdai/llm-agents-from-scratch/issues/new)
   with a clear title and as much context as possible:
   - What you were doing and what you expected to happen
   - What actually happened (error messages, tracebacks, unexpected output)
   - Your Python version, OS, and relevant package versions
   - A minimal reproducible example if possible

---

## Community Projects

Built a new tool integration, a custom agent, or an interesting notebook using
this framework? We'd love to see it.

Share your project on
[GitHub Discussions](https://github.com/nerdai/llm-agents-from-scratch/discussions)
in the **Show and Tell** category. Include a short description of what you
built and a link to your code or notebook.

A dedicated community showcase page is in the works — projects shared in
Discussions may be featured there.

---

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- [Ollama](https://ollama.com/) running locally (default LLM backend)

### Install

```sh
git clone git@github.com:nerdai/llm-agents-from-scratch.git
cd llm-agents-from-scratch
uv sync --all-extras --dev
pre-commit install
```

### Common Commands

```sh
# Run all tests
make test

# Lint (ruff, mypy)
make lint

# Format (ruff)
make format

# Coverage
make coverage-report
```

---

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`.
2. Make your changes, keeping them focused and minimal.
3. Ensure all tests pass (`make test`) and linting is clean (`make lint`).
4. Open a pull request with a clear description of the change and why it's
   needed.

### Code Style

- Line length: 80 characters
- Docstrings: Google style
- Type hints: required throughout (mypy strict mode)
- Imports: sorted by ruff (isort rules, black profile)

These are all enforced automatically by the pre-commit hooks.

---

## Questions

For general questions about the book or the framework, use
[GitHub Discussions](https://github.com/nerdai/llm-agents-from-scratch/discussions).
