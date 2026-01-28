# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**llm-agents-from-scratch** is a Python library for building LLM agents from the ground up, designed to accompany a technical book on creating intelligent agents with Model Context Protocol (MCP) integration. It provides abstract base classes and concrete implementations for LLMs, tools, and agents.

## Commands

```bash
# Install dependencies
uv sync --all-extras --dev

# Run all tests
make test                    # or: pytest tests -v --capture=no

# Run a single test file
pytest tests/test_file.py -v --capture=no

# Run a specific test
pytest tests/test_file.py::test_function_name -v

# Lint and format (runs pre-commit hooks: ruff, mypy, isort, codespell)
make lint

# Format only (black)
make format

# Coverage
make coverage                # Generate XML report
make coverage-report         # Terminal summary
make coverage-html           # HTML report in htmlcov/

# Generate UML diagrams
make diagrams
```

## Architecture

### Core Abstractions

The library uses abstract base classes with type aliases for flexibility:

- **`BaseLLM`** (`base/llm.py`) → Type alias `LLM`
  - Abstract methods: `complete()`, `structured_output()`, `chat()`, `continue_chat_with_tool_results()`
  - Implementation: `OllamaLLM` (primary), `OpenAILLM` (optional extra)

- **`BaseTool` / `AsyncBaseTool`** (`base/tool.py`) → Type alias `Tool = BaseTool | AsyncBaseTool`
  - Required properties: `name`, `description`, `parameters_json_schema`
  - Implementations:
    - `SimpleFunctionTool` / `AsyncSimpleFunctionTool` - Direct function wrapping with JSON schema generation
    - `PydanticFunctionTool` / `AsyncPydanticFunctionTool` - Type-safe functions using Pydantic models
    - `MCPTool` / `MCPToolProvider` - Model Context Protocol integration

### Agent Layer

- **`LLMAgent`** (`agent/llm_agent.py`) - Main agent class managing LLM and tool registry
  - Contains nested `TaskHandler` class extending `asyncio.Future` for async task execution
  - Step-based execution with rollout tracking

- **`LLMAgentBuilder`** (`agent/builder.py`) - Fluent builder pattern for agent construction

### Data Structures (Pydantic Models)

All in `data_structures/`:
- **Task pipeline**: `Task` → `TaskStep` → `TaskStepResult` → `TaskResult`
- **LLM communication**: `ChatMessage`, `ChatRole`, `CompleteResult`
- **Tool invocation**: `ToolCall`, `ToolCallResult`
- **Decision making**: `NextStepDecision`

### Error Hierarchy

Base: `LLMAgentsFromScratchError` (`errors/core.py`)
- `LLMAgentError` → `LLMAgentBuilderError`, `MaxStepsReachedError`
- `MCPError` → `MissingMCPServerParamsError`
- `TaskHandlerError`
- `MissingExtraError`

## Code Style

- **Line length**: 80 characters
- **Docstrings**: Google style (enforced by ruff)
- **Type hints**: Required throughout (mypy strict mode with `disallow_untyped_defs`)
- **Imports**: Sorted with isort (black profile, line-length 79)

## Testing

- Uses pytest with `pytest-asyncio` (asyncio_mode=auto in pytest.ini)
- Tests in `tests/` directory mirror source structure
- Docstrings not required in test files (ruff D rules ignored)
