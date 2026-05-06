import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_agents_from_scratch.agent import LLMAgent
from llm_agents_from_scratch.agent.templates import default_templates
from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ChatRole,
    NextStepDecision,
    Task,
    TaskResult,
    TaskStep,
    TaskStepResult,
    ToolCall,
)
from llm_agents_from_scratch.data_structures.skill import SkillScope
from llm_agents_from_scratch.errors import TaskHandlerError
from llm_agents_from_scratch.skills.skill import Skill
from llm_agents_from_scratch.tools.simple_function import (
    AsyncSimpleFunctionTool,
    SimpleFunctionTool,
)


def test_task_handler_init(
    mock_llm: BaseLLM,
) -> None:
    llm_agent = LLMAgent(
        llm=mock_llm,
    )
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=Task(instruction="mock instruction"),
    )

    assert handler.task.instruction == "mock instruction"
    assert handler.llm_agent == llm_agent


def test_task_handler_init_discovers_skills(
    mock_llm: BaseLLM,
) -> None:
    mock_skill = MagicMock()
    mock_skills = {"my-skill": mock_skill}
    llm_agent = LLMAgent(llm=mock_llm)

    with patch(
        "llm_agents_from_scratch.agent.llm_agent.discover_skills",
        return_value=mock_skills,
    ) as mock_discover:
        handler = LLMAgent.TaskHandler(
            llm_agent=llm_agent,
            task=Task(instruction="mock instruction"),
            skills_scopes=[SkillScope.PROJECT],
        )

    mock_discover.assert_called_once_with([SkillScope.PROJECT])
    assert handler.skills == mock_skills


def test_task_handler_raises_error_when_getting_unset_bg_task(
    mock_llm: BaseLLM,
) -> None:
    llm_agent = LLMAgent(
        llm=mock_llm,
    )
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=Task(instruction="mock instruction"),
    )

    with pytest.raises(TaskHandlerError):
        handler.background_task  # noqa: B018


@pytest.mark.asyncio
async def test_task_handler_raises_error_when_setting_already_set_bg_task(
    mock_llm: BaseLLM,
) -> None:
    async def fn() -> None:
        await asyncio.sleep(0.1)

    llm_agent = LLMAgent(
        llm=mock_llm,
    )
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=Task(instruction="mock instruction"),
    )

    handler.background_task = asyncio.create_task(fn())
    with pytest.raises(TaskHandlerError):
        new_task = asyncio.create_task(fn())
        handler.background_task = new_task

    # cleanup
    handler.background_task.cancel()
    new_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await handler.background_task
        await new_task


@pytest.mark.asyncio
async def test_get_next_step(mock_llm: BaseLLM) -> None:
    """Tests get next step."""

    task = Task(instruction="mock instruction")
    llm_agent = LLMAgent(
        llm=mock_llm,
    )
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=task,
    )

    # initial task step
    initial_step = await handler.get_next_step(previous_step_result=None)

    # update rollout and get next step
    expected_next_step = NextStepDecision(
        kind="next_step",
        content="Some next instruction.",
    )

    magic_mock_llm = AsyncMock()
    magic_mock_llm.structured_output.return_value = expected_next_step
    handler.llm_agent.llm = magic_mock_llm
    handler.rollout = "some progress"
    next_step = await handler.get_next_step(
        previous_step_result=TaskStepResult(
            task_step_id=initial_step.id_,
            content="mock step result",
        ),
    )

    assert initial_step.instruction == "mock instruction"
    assert isinstance(next_step, TaskStep)
    assert next_step.instruction == expected_next_step.content


@pytest.mark.asyncio
async def test_get_next_step_completes_task(mock_llm: BaseLLM) -> None:
    """Tests get next step returns TaskResult."""
    task = Task(instruction="mock instruction")
    llm_agent = LLMAgent(
        llm=mock_llm,
    )
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=task,
    )

    # initial task step
    initial_step = await handler.get_next_step(previous_step_result=None)

    # update rollout and get next step
    expected_next_step = NextStepDecision(
        kind="final_result",
        content="",
    )

    magic_mock_llm = AsyncMock()
    magic_mock_llm.structured_output.return_value = expected_next_step
    handler.llm_agent.llm = magic_mock_llm
    handler.rollout = "some progress"
    next_step = await handler.get_next_step(
        previous_step_result=TaskStepResult(
            task_step_id=initial_step.id_,
            content="sufficient result",
        ),
    )

    assert initial_step.instruction == "mock instruction"
    assert isinstance(next_step, TaskResult)
    assert next_step.content == "sufficient result"  # prev step result content


@pytest.mark.asyncio
async def test_get_next_step_raises_error_from_structured_output_call(
    mock_llm: BaseLLM,
) -> None:
    """Tests get next step raises error when invoking ~llm.structured_output."""

    task = Task(instruction="mock instruction")
    llm_agent = LLMAgent(
        llm=mock_llm,
    )
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=task,
    )

    # initial task step
    initial_step = await handler.get_next_step(previous_step_result=None)

    # update rollout and get next step
    magic_mock_llm = AsyncMock()
    magic_mock_llm.structured_output.side_effect = RuntimeError("oops.")
    handler.llm_agent.llm = magic_mock_llm
    handler.rollout = "some progress"

    with pytest.raises(
        TaskHandlerError,
        match="Failed to get next step: oops.",
    ):
        await handler.get_next_step(
            previous_step_result=TaskStepResult(
                task_step_id=initial_step.id_,
                content="mock step result",
            ),
        )

    assert initial_step.instruction == "mock instruction"


def test_private_format_step_for_rollout(
    mock_llm: BaseLLM,
) -> None:
    """Tests helper method to get rollout contribution from run step."""
    task = Task(instruction="mock instruction")
    llm_agent = LLMAgent(
        llm=mock_llm,
    )
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=task,
    )
    a_tool_call = ToolCall(
        tool_name="a tool",
        arguments={"tool_arg": 1},
    )
    chat_history = [
        ChatMessage(
            role=ChatRole.SYSTEM,
            content="a system message",
        ),
        ChatMessage(
            role=ChatRole.USER,
            content="a user message",
        ),
        ChatMessage(
            role=ChatRole.ASSISTANT,
            content="",
            tool_calls=[
                a_tool_call,
            ],
        ),
        ChatMessage(
            role=ChatRole.TOOL,
            content="\n\ttool name: `a tool`\n\ttool result: 1+2=3.",
        ),
        ChatMessage(
            role=ChatRole.ASSISTANT,
            content="done!",
        ),
    ]

    # act
    rollout_contribution = handler._format_step_for_rollout(
        chat_history=chat_history,
    )

    expected_rollout_contribution = (
        "=== Task Step Start ===\n\n"
        "💬 assistant: My current instruction is 'a user message'\n\n"
        "💬 assistant: I need to make the following tool call(s):"
        f"\n\n{a_tool_call.model_dump_json(indent=4)}.\n\n"
        "🔧 tool: \n\ttool name: `a tool`\n\ttool result: 1+2=3.\n\n"
        "💬 assistant: done!\n\n"
        "=== Task Step End ==="
    )

    assert rollout_contribution == expected_rollout_contribution


@pytest.mark.asyncio
async def test_run_step() -> None:
    """Tests run step."""

    def plus_one(arg1: int) -> int:
        return arg1 + 1

    # async simple tool
    async def plus_two(arg1: int) -> int:
        await asyncio.sleep(0.1)
        return arg1 + 2

    # arrange mocks
    mock_llm = AsyncMock()
    # initial chat response
    tool_calls = [
        ToolCall(
            tool_name="plus_one",
            arguments={"arg1": 1},
        ),
        ToolCall(
            tool_name="plus_two",
            arguments={"arg1": 1},
        ),
        # this tool doesn't exist
        ToolCall(
            tool_name="plus_three",
            arguments={"arg1": 1},
        ),
    ]
    mock_llm.chat.return_value = (
        ChatMessage(
            role=ChatRole.USER,
            content="Some instruction.",
        ),
        ChatMessage(
            role=ChatRole.ASSISTANT,
            content="Initial response.",
            tool_calls=tool_calls,
        ),
    )
    # continue conversation with tool calls
    mock_return_value = (
        [
            # tool calls
            ChatMessage(
                role=ChatRole.TOOL,
                content="2",
            ),
            ChatMessage(
                role=ChatRole.TOOL,
                content="3",
            ),
            ChatMessage(
                role=ChatRole.TOOL,
                content="error: tool name `plus_three` doesn't exist",
            ),
        ],
        ChatMessage(
            role=ChatRole.ASSISTANT,
            content="The final response.",
        ),
    )
    mock_llm.continue_chat_with_tool_results.return_value = mock_return_value

    task = Task(instruction="mock instruction")
    llm_agent = LLMAgent(
        llm=mock_llm,
        tools=[
            SimpleFunctionTool(func=plus_one),
            AsyncSimpleFunctionTool(func=plus_two),
        ],
    )
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=task,
    )

    # act
    step = TaskStep(
        task_id=task.id_,
        instruction="Some instruction.",
    )
    step_result = await handler.run_step(step)

    # assert
    mock_llm.chat.assert_awaited_once_with(
        input="Some instruction.",
        chat_history=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=default_templates[
                    "run_step_system_message_without_rollout"
                ].format(
                    llm_agent_system_message=llm_agent.templates[
                        "system_message"
                    ],
                    current_rollout="",
                ),
            ),
        ],
        tools=list(handler.llm_agent.tools_registry.values()),
    )
    mock_llm.continue_chat_with_tool_results.assert_awaited_once()
    assert step_result.task_step_id == step.id_
    assert step_result.content == "The final response."


@pytest.mark.asyncio
async def test_run_step_without_tool_calls() -> None:
    """Tests run step."""

    # arrange mocks
    mock_llm = AsyncMock()
    mock_llm.chat.return_value = (
        ChatMessage(
            role=ChatRole.USER,
            content="Some instruction.",
        ),
        ChatMessage(
            role=ChatRole.ASSISTANT,
            content="Initial response.",
        ),
    )

    llm_agent = LLMAgent(
        llm=mock_llm,
    )
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=Task(instruction="mock instruction"),
    )

    # act
    step = TaskStep(
        task_id=handler.task.id_,
        instruction="Some instruction.",
        last_step=False,
    )
    step_result = await handler.run_step(step)

    # assert
    mock_llm.chat.assert_awaited_once_with(
        input="Some instruction.",
        chat_history=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=default_templates[
                    "run_step_system_message_without_rollout"
                ].format(
                    llm_agent_system_message=llm_agent.templates[
                        "system_message"
                    ],
                    current_rollout="",
                ),
            ),
        ],
        tools=list(handler.llm_agent.tools_registry.values()),
    )
    mock_llm.continue_chat_with_tool_results.assert_not_awaited()
    assert step_result.task_step_id == step.id_
    assert step_result.content == "Initial response."
    assert str(step_result) == "Initial response."


@pytest.mark.asyncio
async def test_run_step_with_tool_calls_in_final_response() -> None:
    """Tests run step."""

    def plus_one(arg1: int) -> int:
        return arg1 + 1

    # async simple tool
    async def plus_two(arg1: int) -> int:
        await asyncio.sleep(0.1)
        return arg1 + 2

    # arrange mocks
    mock_llm = AsyncMock()
    # initial chat response
    tool_calls = [
        ToolCall(
            tool_name="plus_one",
            arguments={"arg1": 1},
        ),
        ToolCall(
            tool_name="plus_two",
            arguments={"arg1": 1},
        ),
        # this tool doesn't exist
        ToolCall(
            tool_name="plus_three",
            arguments={"arg1": 1},
        ),
    ]
    mock_llm.chat.return_value = (
        ChatMessage(
            role=ChatRole.USER,
            content="Some instruction.",
        ),
        ChatMessage(
            role=ChatRole.ASSISTANT,
            content="Initial response.",
            tool_calls=tool_calls,
        ),
    )
    # continue conversation with tool calls
    # this will return another set tool call request
    second_set_tool_calls = [
        ToolCall(
            tool_name="plus_one",
            arguments={"arg1": 2},
        ),
        ToolCall(
            tool_name="plus_two",
            arguments={"arg1": 2},
        ),
    ]
    mock_return_value = (
        [
            # tool calls
            ChatMessage(
                role=ChatRole.TOOL,
                content="2",
            ),
            ChatMessage(
                role=ChatRole.TOOL,
                content="3",
            ),
            ChatMessage(
                role=ChatRole.TOOL,
                content="error: tool name `plus_three` doesn't exist",
            ),
        ],
        ChatMessage(
            role=ChatRole.ASSISTANT,
            content="",
            # final response contains more tool calls
            tool_calls=second_set_tool_calls,
        ),
    )
    mock_llm.continue_chat_with_tool_results.return_value = mock_return_value

    task = Task(instruction="mock instruction")
    llm_agent = LLMAgent(
        llm=mock_llm,
        tools=[
            SimpleFunctionTool(func=plus_one),
            AsyncSimpleFunctionTool(func=plus_two),
        ],
    )
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=task,
    )

    # act
    step = TaskStep(
        task_id=task.id_,
        instruction="Some instruction.",
    )
    step_result = await handler.run_step(step)

    # assert
    mock_llm.chat.assert_awaited_once_with(
        input="Some instruction.",
        chat_history=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=default_templates[
                    "run_step_system_message_without_rollout"
                ].format(
                    llm_agent_system_message=llm_agent.templates[
                        "system_message"
                    ],
                    current_rollout="",
                ),
            ),
        ],
        tools=list(handler.llm_agent.tools_registry.values()),
    )
    mock_llm.continue_chat_with_tool_results.assert_awaited_once()
    assert step_result.task_step_id == step.id_
    expected_final_content = (
        "I need to make the following tool-calls:\n"
        + "\n".join(t.model_dump_json(indent=4) for t in second_set_tool_calls)
    )
    assert step_result.content == expected_final_content


def test_task_handler_use_skill_tool_set_when_skills_present(
    mock_llm: BaseLLM,
) -> None:
    """Tests _use_skill_tool is set when skills are discovered."""
    mock_skill = MagicMock(spec=Skill)

    llm_agent = LLMAgent(llm=mock_llm)
    with patch(
        "llm_agents_from_scratch.agent.llm_agent.discover_skills",
        return_value={"my-skill": mock_skill},
    ):
        handler = LLMAgent.TaskHandler(
            llm_agent=llm_agent,
            task=Task(instruction="mock instruction"),
        )

    assert handler._use_skill_tool is not None
    assert handler._use_skill_tool.name == "from_scratch__use_skill"


def test_task_handler_use_skill_tool_none_when_no_skills(
    mock_llm: BaseLLM,
) -> None:
    """Tests _use_skill_tool is None when no skills are discovered."""
    llm_agent = LLMAgent(llm=mock_llm)
    with patch(
        "llm_agents_from_scratch.agent.llm_agent.discover_skills",
        return_value={},
    ):
        handler = LLMAgent.TaskHandler(
            llm_agent=llm_agent,
            task=Task(instruction="mock instruction"),
        )

    assert handler._use_skill_tool is None


def test_skills_catalog_empty_when_no_skills(mock_llm: BaseLLM) -> None:
    """Tests _skills_catalog returns empty string when no skills."""
    llm_agent = LLMAgent(llm=mock_llm)
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=Task(instruction="mock instruction"),
    )
    assert handler._skills_catalog == ""


def test_skills_catalog_returns_catalog_xml(mock_llm: BaseLLM) -> None:
    """Tests _skills_catalog returns formatted XML when skills present."""
    mock_skill = MagicMock(spec=Skill)
    mock_skill.catalog.return_value = "<skill><name>my-skill</name></skill>"

    llm_agent = LLMAgent(llm=mock_llm)
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=Task(instruction="mock instruction"),
    )
    handler.skills = {"my-skill": mock_skill}

    expected = default_templates["skills_catalog"].format(
        skills="<skill><name>my-skill</name></skill>",
    )
    assert handler._skills_catalog == expected


def test_skills_catalog_excludes_explicit_only_skills(
    mock_llm: BaseLLM,
) -> None:
    """Tests _skills_catalog omits skills in _explicit_only_skills."""
    mock_skill = MagicMock(spec=Skill)

    llm_agent = LLMAgent(llm=mock_llm)
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=Task(instruction="mock instruction"),
    )
    handler.skills = {"my-skill": mock_skill}
    handler._explicit_only_skills = {"my-skill"}

    assert handler._skills_catalog == ""
    mock_skill.catalog.assert_not_called()


@pytest.mark.asyncio
async def test_run_step_injects_skills_catalog() -> None:
    """Tests run_step appends skills catalog to system message when present."""
    mock_skill = MagicMock(spec=Skill)
    mock_skill.catalog.return_value = "<skill><name>my-skill</name></skill>"

    mock_llm = AsyncMock()
    mock_llm.chat.return_value = (
        ChatMessage(role=ChatRole.USER, content="Some instruction."),
        ChatMessage(role=ChatRole.ASSISTANT, content="Done."),
    )

    llm_agent = LLMAgent(llm=mock_llm)
    handler = LLMAgent.TaskHandler(
        llm_agent=llm_agent,
        task=Task(instruction="mock instruction"),
    )
    handler.skills = {"my-skill": mock_skill}

    step = TaskStep(
        task_id=handler.task.id_,
        instruction="Some instruction.",
    )
    await handler.run_step(step)

    expected_catalog = default_templates["skills_catalog"].format(
        skills="<skill><name>my-skill</name></skill>",
    )
    expected_system_content = (
        default_templates["run_step_system_message_without_rollout"].format(
            llm_agent_system_message=llm_agent.templates["system_message"],
        )
        + f"\n\n{expected_catalog}"
    )
    mock_llm.chat.assert_awaited_once_with(
        input="Some instruction.",
        chat_history=[
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=expected_system_content,
            ),
        ],
        tools=list(handler.llm_agent.tools_registry.values()),
    )
