"""Unit tests for UseSubAgentTool."""

import asyncio
import json
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_agents_from_scratch.agent import LLMAgent, LLMAgentBuilder
from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures import Task, TaskResult, ToolCall
from llm_agents_from_scratch.data_structures.skill import SkillScope
from llm_agents_from_scratch.errors import MaxStepsReachedError
from llm_agents_from_scratch.logger import current_subagent_name
from llm_agents_from_scratch.subagents import SubAgentSpec, UseSubAgentTool
from llm_agents_from_scratch.tools.mcp.tool import MCPTool

MAX_STEPS_CODER = 5
DISPATCH_COUNT = 2


def make_spec(
    name: str,
    mock_llm: BaseLLM,
    max_steps: int | None = None,
    skills_scopes: list[SkillScope] | None = None,
    explicit_only_skills: set[str] | None = None,
) -> SubAgentSpec:
    return SubAgentSpec(
        name=name,
        description=f"Sub-agent: {name}",
        builder=LLMAgentBuilder(llm=mock_llm),
        max_steps=max_steps,
        skills_scopes=skills_scopes,
        explicit_only_skills=explicit_only_skills,
    )


@contextmanager
def patched_dispatch(
    spec: SubAgentSpec,
    mock_llm: BaseLLM,
    **run_kwargs: object,
) -> Iterator[tuple[LLMAgent, MagicMock]]:
    """Patches spec.builder.build() to return one pre-built LLMAgent for
    the duration of this context, and patches that agent's run() per
    run_kwargs (forwarded to patch.object).

    Not for tests exercising multiple dispatches in one context: build()
    returns the *same* agent instance on every call here, not a fresh
    one per dispatch -- see
    test_use_subagent_tool_builds_fresh_agent_per_dispatch for that.
    """
    agent = LLMAgent(llm=mock_llm)
    with (
        patch.object(spec.builder, "build", AsyncMock(return_value=agent)),
        patch.object(agent, "run", **run_kwargs) as mock_run,  # type: ignore[arg-type]
    ):
        yield agent, mock_run


def test_use_subagent_tool_name(mock_llm: BaseLLM) -> None:
    """Tests UseSubAgentTool.name."""
    tool = UseSubAgentTool(subagents_registry={})
    assert tool.name == "from_scratch__use_subagent"


def test_use_subagent_tool_description(mock_llm: BaseLLM) -> None:
    """Tests UseSubAgentTool.description mentions dispatch."""
    tool = UseSubAgentTool(subagents_registry={})
    assert "dispatch" in tool.description.lower()


def test_use_subagent_tool_schema_enum(mock_llm: BaseLLM) -> None:
    """Tests parameters_json_schema enum contains registered subagent names."""
    subagents = {
        "researcher": make_spec("researcher", mock_llm),
        "coder": make_spec("coder", mock_llm),
    }
    tool = UseSubAgentTool(subagents_registry=subagents)
    enum = tool.parameters_json_schema["properties"]["name"]["enum"]
    assert "researcher" in enum
    assert "coder" in enum


def test_use_subagent_tool_schema_required(mock_llm: BaseLLM) -> None:
    """Tests parameters_json_schema requires both name and task."""
    tool = UseSubAgentTool(subagents_registry={})
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
    with patched_dispatch(
        spec,
        mock_llm,
        return_value=future,
    ) as (_agent, mock_run):
        tool = UseSubAgentTool(subagents_registry={"researcher": spec})
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
    assert call_args.kwargs.get("skills_scopes") is None
    assert call_args.kwargs.get("explicit_only_skills") is None


@pytest.mark.asyncio
async def test_use_subagent_tool_passes_max_steps(mock_llm: BaseLLM) -> None:
    """Tests max_steps from SubAgentSpec is forwarded to agent.run()."""
    future: asyncio.Future[TaskResult] = (
        asyncio.get_running_loop().create_future()
    )
    future.set_result(TaskResult(task_id="t1", content="done"))

    spec = make_spec("coder", mock_llm, max_steps=MAX_STEPS_CODER)
    with patched_dispatch(spec, mock_llm, return_value=future) as (
        _agent,
        mock_run,
    ):
        tool = UseSubAgentTool(subagents_registry={"coder": spec})
        tool_call = ToolCall(
            tool_name="from_scratch__use_subagent",
            arguments={"name": "coder", "task": "Write a sort function."},
        )
        await tool(tool_call=tool_call)

    assert mock_run.call_args.kwargs.get("max_steps") == MAX_STEPS_CODER


@pytest.mark.asyncio
async def test_use_subagent_tool_passes_skills_scopes(
    mock_llm: BaseLLM,
) -> None:
    """Tests skills_scopes from SubAgentSpec is forwarded to agent.run()."""
    future: asyncio.Future[TaskResult] = (
        asyncio.get_running_loop().create_future()
    )
    future.set_result(TaskResult(task_id="t1", content="done"))

    spec = make_spec(
        "coder",
        mock_llm,
        skills_scopes=[SkillScope.PROJECT],
    )
    with patched_dispatch(spec, mock_llm, return_value=future) as (
        _agent,
        mock_run,
    ):
        tool = UseSubAgentTool(subagents_registry={"coder": spec})
        tool_call = ToolCall(
            tool_name="from_scratch__use_subagent",
            arguments={"name": "coder", "task": "Write a sort function."},
        )
        await tool(tool_call=tool_call)

    assert mock_run.call_args.kwargs.get("skills_scopes") == [
        SkillScope.PROJECT,
    ]


@pytest.mark.asyncio
async def test_use_subagent_tool_passes_explicit_only_skills(
    mock_llm: BaseLLM,
) -> None:
    """Tests explicit_only_skills is forwarded to agent.run()."""
    future: asyncio.Future[TaskResult] = (
        asyncio.get_running_loop().create_future()
    )
    future.set_result(TaskResult(task_id="t1", content="done"))

    spec = make_spec(
        "coder",
        mock_llm,
        explicit_only_skills={"stop-at-one"},
    )
    with patched_dispatch(spec, mock_llm, return_value=future) as (
        _agent,
        mock_run,
    ):
        tool = UseSubAgentTool(subagents_registry={"coder": spec})
        tool_call = ToolCall(
            tool_name="from_scratch__use_subagent",
            arguments={"name": "coder", "task": "Write a sort function."},
        )
        await tool(tool_call=tool_call)

    assert mock_run.call_args.kwargs.get("explicit_only_skills") == {
        "stop-at-one",
    }


@pytest.mark.asyncio
async def test_use_subagent_tool_sets_current_subagent_name_during_run(
    mock_llm: BaseLLM,
) -> None:
    """Tests current_subagent_name is set while agent.run() executes."""
    future: asyncio.Future[TaskResult] = (
        asyncio.get_running_loop().create_future()
    )
    future.set_result(TaskResult(task_id="t1", content="done"))
    observed: dict[str, str | None] = {}

    def fake_run(*args: object, **kwargs: object) -> asyncio.Future[TaskResult]:
        observed["name"] = current_subagent_name.get()
        return future

    spec = make_spec("researcher", mock_llm)
    with patched_dispatch(spec, mock_llm, side_effect=fake_run):
        tool = UseSubAgentTool(subagents_registry={"researcher": spec})
        tool_call = ToolCall(
            tool_name="from_scratch__use_subagent",
            arguments={"name": "researcher", "task": "do it"},
        )
        await tool(tool_call=tool_call)

    assert observed["name"] == "researcher"
    assert current_subagent_name.get() is None


@pytest.mark.asyncio
async def test_use_subagent_tool_resets_current_subagent_name_on_error(
    mock_llm: BaseLLM,
) -> None:
    """Tests current_subagent_name resets even when the sub-agent raises."""
    future: asyncio.Future[TaskResult] = (
        asyncio.get_running_loop().create_future()
    )
    future.set_exception(RuntimeError("boom"))

    spec = make_spec("coder", mock_llm)
    with patched_dispatch(spec, mock_llm, return_value=future):
        tool = UseSubAgentTool(subagents_registry={"coder": spec})
        tool_call = ToolCall(
            tool_name="from_scratch__use_subagent",
            arguments={"name": "coder", "task": "do it"},
        )
        await tool(tool_call=tool_call)

    assert current_subagent_name.get() is None


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
    with patched_dispatch(spec, mock_llm, return_value=future):
        tool = UseSubAgentTool(subagents_registry={"researcher": spec})
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
    with patched_dispatch(spec, mock_llm, return_value=future):
        tool = UseSubAgentTool(subagents_registry={"coder": spec})
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
async def test_use_subagent_tool_catches_build_error(mock_llm: BaseLLM) -> None:
    """Tests a builder.build() failure is caught as an error result too.

    No dedicated error type for build failures -- they fall through to
    the same catch as dispatch/run failures.
    """
    spec = make_spec("coder", mock_llm)
    with patch.object(
        spec.builder,
        "build",
        AsyncMock(side_effect=RuntimeError("bad config")),
    ):
        tool = UseSubAgentTool(subagents_registry={"coder": spec})
        tool_call = ToolCall(
            tool_name="from_scratch__use_subagent",
            arguments={"name": "coder", "task": "Do something."},
        )
        result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "RuntimeError"
    assert details["subagent"] == "coder"
    assert "bad config" in details["message"]
    assert current_subagent_name.get() is None


@pytest.mark.asyncio
async def test_use_subagent_tool_unknown_name(mock_llm: BaseLLM) -> None:
    """Tests unknown subagent name returns error result.

    Caught by the ``name`` enum in ``parameters_json_schema``, which
    lists the valid subagent names directly in the ``ValidationError``
    message.
    """
    tool = UseSubAgentTool(
        subagents_registry={"researcher": make_spec("researcher", mock_llm)},
    )
    tool_call = ToolCall(
        tool_name="from_scratch__use_subagent",
        arguments={"name": "unknown", "task": "Do something."},
    )
    result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert details["error_type"] == "ValidationError"
    assert "unknown" in details["message"]


@pytest.mark.asyncio
async def test_use_subagent_tool_missing_name_arg(mock_llm: BaseLLM) -> None:
    """Tests missing name argument returns error result."""
    tool = UseSubAgentTool(subagents_registry={})
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
    tool = UseSubAgentTool(subagents_registry={})
    tool_call = ToolCall(
        tool_name="from_scratch__use_subagent",
        arguments={"name": "researcher"},
    )
    result = await tool(tool_call=tool_call)

    assert result.error is True
    details = json.loads(result.content)
    assert "'task'" in details["message"]


@pytest.mark.asyncio
async def test_use_subagent_tool_builds_fresh_agent_per_dispatch(
    mock_llm: BaseLLM,
) -> None:
    """Tests each dispatch gets a distinct LLMAgent built from the spec.

    The precise claim behind the builder-recipe design: build() is
    called fresh per dispatch (a new LLMAgent shell each time), not
    reused as a live singleton the way the pre-refactor `agent` field
    was.
    """
    spec = make_spec("coder", mock_llm)
    built_agents: list[LLMAgent] = []
    real_build = spec.builder.build

    async def build_and_stub_run() -> LLMAgent:
        agent = await real_build()
        built_agents.append(agent)
        future: asyncio.Future[TaskResult] = (
            asyncio.get_running_loop().create_future()
        )
        future.set_result(TaskResult(task_id="t", content="done"))
        agent.run = MagicMock(return_value=future)  # type: ignore[method-assign]
        return agent

    with patch.object(spec.builder, "build", side_effect=build_and_stub_run):
        tool = UseSubAgentTool(subagents_registry={"coder": spec})
        for _ in range(2):
            tool_call = ToolCall(
                tool_name="from_scratch__use_subagent",
                arguments={"name": "coder", "task": "do it"},
            )
            await tool(tool_call=tool_call)

    assert len(built_agents) == DISPATCH_COUNT
    assert built_agents[0] is not built_agents[1]


@pytest.mark.asyncio
async def test_use_subagent_tool_reuses_mcp_provider_across_dispatches(
    mock_llm: BaseLLM,
) -> None:
    """Tests the same MCPToolProvider is reused across dispatches.

    The key claim behind holding a builder rather than a built agent:
    resources the builder was given directly (here, an MCPToolProvider
    and the persistent session it owns) are reused as-is across every
    build() call -- only the LLMAgent shell is rebuilt.
    """
    provider = MagicMock()
    mcp_tool = MCPTool(
        provider=provider,
        name="mcp__demo__ping",
        desc="Pings the demo server.",
        params_json_schema={"type": "object", "properties": {}},
    )
    provider.get_tools = AsyncMock(return_value=[mcp_tool])

    builder = LLMAgentBuilder(llm=mock_llm, mcp_providers=[provider])
    spec = SubAgentSpec(
        name="coder",
        description="Writes code.",
        builder=builder,
    )

    built_agents: list[LLMAgent] = []
    real_build = builder.build

    async def build_and_stub_run() -> LLMAgent:
        agent = await real_build()
        built_agents.append(agent)
        future: asyncio.Future[TaskResult] = (
            asyncio.get_running_loop().create_future()
        )
        future.set_result(TaskResult(task_id="t", content="done"))
        agent.run = MagicMock(return_value=future)  # type: ignore[method-assign]
        return agent

    with patch.object(builder, "build", side_effect=build_and_stub_run):
        tool = UseSubAgentTool(subagents_registry={"coder": spec})
        for _ in range(2):
            tool_call = ToolCall(
                tool_name="from_scratch__use_subagent",
                arguments={"name": "coder", "task": "do it"},
            )
            await tool(tool_call=tool_call)

    assert len(built_agents) == DISPATCH_COUNT
    agent_a, agent_b = built_agents
    assert agent_a is not agent_b
    assert (
        agent_a.tools_registry["mcp__demo__ping"].provider  # type: ignore[union-attr]
        is agent_b.tools_registry["mcp__demo__ping"].provider  # type: ignore[union-attr]
        is provider
    )
    assert provider.get_tools.await_count == DISPATCH_COUNT
