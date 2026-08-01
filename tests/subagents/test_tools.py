"""Unit tests for UseSubAgentTool."""

import asyncio
import json
from unittest.mock import patch

import pytest

from llm_agents_from_scratch.agent import LLMAgent
from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures import Task, TaskResult, ToolCall
from llm_agents_from_scratch.errors import MaxStepsReachedError
from llm_agents_from_scratch.subagents import SubAgentSpec, UseSubAgentTool

MAX_STEPS_CODER = 5


def make_spec(
    name: str,
    mock_llm: BaseLLM,
    max_steps: int | None = None,
) -> SubAgentSpec:
    return SubAgentSpec(
        name=name,
        description=f"Sub-agent: {name}",
        agent=LLMAgent(llm=mock_llm),
        max_steps=max_steps,
    )


def test_use_subagent_tool_name(mock_llm: BaseLLM) -> None:
    """Tests UseSubAgentTool.name."""
    tool = UseSubAgentTool(subagents={})
    assert tool.name == "from_scratch__use_subagent"


def test_use_subagent_tool_description(mock_llm: BaseLLM) -> None:
    """Tests UseSubAgentTool.description mentions dispatch."""
    tool = UseSubAgentTool(subagents={})
    assert "dispatch" in tool.description.lower()


def test_use_subagent_tool_schema_enum(mock_llm: BaseLLM) -> None:
    """Tests parameters_json_schema enum contains registered subagent names."""
    subagents = {
        "researcher": make_spec("researcher", mock_llm),
        "coder": make_spec("coder", mock_llm),
    }
    tool = UseSubAgentTool(subagents=subagents)
    enum = tool.parameters_json_schema["properties"]["name"]["enum"]
    assert "researcher" in enum
    assert "coder" in enum


def test_use_subagent_tool_schema_required(mock_llm: BaseLLM) -> None:
    """Tests parameters_json_schema requires both name and task."""
    tool = UseSubAgentTool(subagents={})
    required = tool.parameters_json_schema["required"]
    assert "name" in required
    assert "task" in required


@pytest.mark.asyncio
async def test_use_subagent_tool_dispatches_task(mock_llm: BaseLLM) -> None:
    """Tests successful dispatch returns sub-agent result.content."""
    task_result = TaskResult(task_id="t1", content="42 is the answer")
    future: asyncio.Future[TaskResult] = (
        asyncio.get_running_loop().create_future()
    )
    future.set_result(task_result)

    spec = make_spec("researcher", mock_llm)
    with patch.object(spec.agent, "run", return_value=future) as mock_run:
        tool = UseSubAgentTool(subagents={"researcher": spec})
        tool_call = ToolCall(
            tool_name="from_scratch__use_subagent",
            arguments={"name": "researcher", "task": "What is the answer?"},
        )
        result = await tool(tool_call=tool_call)

    assert result.error is False
    assert result.content == "42 is the answer"
    mock_run.assert_called_once()
    call_args = mock_run.call_args
    assert isinstance(call_args.args[0], Task)
    assert call_args.args[0].instruction == "What is the answer?"
    assert call_args.kwargs.get("max_steps") is None


@pytest.mark.asyncio
async def test_use_subagent_tool_passes_max_steps(mock_llm: BaseLLM) -> None:
    """Tests max_steps from SubAgentSpec is forwarded to agent.run()."""
    future: asyncio.Future[TaskResult] = (
        asyncio.get_running_loop().create_future()
    )
    future.set_result(TaskResult(task_id="t1", content="done"))

    spec = make_spec("coder", mock_llm, max_steps=MAX_STEPS_CODER)
    with patch.object(spec.agent, "run", return_value=future) as mock_run:
        tool = UseSubAgentTool(subagents={"coder": spec})
        tool_call = ToolCall(
            tool_name="from_scratch__use_subagent",
            arguments={"name": "coder", "task": "Write a sort function."},
        )
        await tool(tool_call=tool_call)

    assert mock_run.call_args.kwargs.get("max_steps") == MAX_STEPS_CODER


@pytest.mark.asyncio
async def test_use_subagent_tool_catches_max_steps_error(
    mock_llm: BaseLLM,
) -> None:
    """Tests MaxStepsReachedError is caught and returned as error result."""
    future: asyncio.Future[TaskResult] = (
        asyncio.get_running_loop().create_future()
    )
    future.set_exception(MaxStepsReachedError("Max steps reached."))

    spec = make_spec("researcher", mock_llm)
    with patch.object(spec.agent, "run", return_value=future):
        tool = UseSubAgentTool(subagents={"researcher": spec})
        tool_call = ToolCall(
            tool_name="from_scratch__use_subagent",
            arguments={"name": "researcher", "task": "Find everything."},
        )
        result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "MaxStepsReachedError"
    assert details["subagent"] == "researcher"


@pytest.mark.asyncio
async def test_use_subagent_tool_catches_unexpected_error(
    mock_llm: BaseLLM,
) -> None:
    """Tests unexpected sub-agent exceptions are caught as error results."""
    future: asyncio.Future[TaskResult] = (
        asyncio.get_running_loop().create_future()
    )
    future.set_exception(RuntimeError("boom"))

    spec = make_spec("coder", mock_llm)
    with patch.object(spec.agent, "run", return_value=future):
        tool = UseSubAgentTool(subagents={"coder": spec})
        tool_call = ToolCall(
            tool_name="from_scratch__use_subagent",
            arguments={"name": "coder", "task": "Do something."},
        )
        result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "RuntimeError"
    assert details["subagent"] == "coder"
    assert "boom" in details["message"]


@pytest.mark.asyncio
async def test_use_subagent_tool_unknown_name(mock_llm: BaseLLM) -> None:
    """Tests unknown subagent name returns error result."""
    tool = UseSubAgentTool(
        subagents={"researcher": make_spec("researcher", mock_llm)},
    )
    tool_call = ToolCall(
        tool_name="from_scratch__use_subagent",
        arguments={"name": "unknown", "task": "Do something."},
    )
    result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "SubAgentNotFoundError"
    assert "not found" in details["message"]


@pytest.mark.asyncio
async def test_use_subagent_tool_missing_name_arg(mock_llm: BaseLLM) -> None:
    """Tests missing name argument returns error result."""
    tool = UseSubAgentTool(subagents={})
    tool_call = ToolCall(
        tool_name="from_scratch__use_subagent",
        arguments={"task": "Do something."},
    )
    result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert "'name'" in details["message"]


@pytest.mark.asyncio
async def test_use_subagent_tool_missing_task_arg(mock_llm: BaseLLM) -> None:
    """Tests missing task argument returns error result."""
    tool = UseSubAgentTool(subagents={})
    tool_call = ToolCall(
        tool_name="from_scratch__use_subagent",
        arguments={"name": "researcher"},
    )
    result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert "'task'" in details["message"]
