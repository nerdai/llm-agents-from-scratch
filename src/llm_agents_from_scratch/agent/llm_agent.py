"""Agent Module."""

import asyncio
import json
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from typing_extensions import Self

from llm_agents_from_scratch.a2a.client.spec import A2AAgentSpec
from llm_agents_from_scratch.a2a.client.tools import UseA2AAgentTool
from llm_agents_from_scratch.base.llm import LLM
from llm_agents_from_scratch.base.tool import AsyncBaseTool, Tool
from llm_agents_from_scratch.data_structures import (
    ApprovalResult,
    ChatMessage,
    ChatRole,
    NextStepDecision,
    RejectedTaskResult,
    Task,
    TaskResult,
    TaskStep,
    TaskStepResult,
    ToolCall,
    ToolCallResult,
)
from llm_agents_from_scratch.data_structures.memory import Episode
from llm_agents_from_scratch.data_structures.skill import SkillScope
from llm_agents_from_scratch.errors import (
    LLMAgentError,
    MaxStepsReachedError,
    RecordMemoryError,
    TaskHandlerError,
)
from llm_agents_from_scratch.logger import get_logger
from llm_agents_from_scratch.memory.memory import Memory
from llm_agents_from_scratch.skills.constants import (
    EXPLICIT_SKILL_ACTIVATION_TEMPLATE,
    EXPLICIT_SKILL_ACTIVATION_WITH_PROMPT_TEMPLATE,
)
from llm_agents_from_scratch.skills.discovery import discover_skills
from llm_agents_from_scratch.skills.skill import Skill
from llm_agents_from_scratch.skills.tools import UseSkillTool

from .templates import LLMAgentTemplates, default_templates

if TYPE_CHECKING:
    from llm_agents_from_scratch.subagents.spec import SubAgentSpec
    from llm_agents_from_scratch.subagents.tools import UseSubAgentTool


def _prompt_for_approval(task_result: TaskResult) -> ApprovalResult:
    """Render a proposed task result and ask the operator to approve it.

    Added in Chapter 8. Blocking `rich` prompt, run via
    ``asyncio.to_thread`` by ``TaskHandler.request_approval``.

    Args:
        task_result (TaskResult): The proposed task result to review.

    Returns:
        ApprovalResult: The approval decision.

    Raises:
        EOFError: If stdin is closed.
        KeyboardInterrupt: If the operator interrupts.
    """
    console = Console()
    console.print(
        Panel(
            task_result.content,
            title="Proposed Task Result",
            border_style="cyan",
        ),
    )
    approved = Confirm.ask("Approve this result?", console=console)
    if approved:
        return ApprovalResult(approved=True, feedback="")
    feedback = Prompt.ask(
        "Provide your correction rationale for the LLM agent to address",
        console=console,
    )
    return ApprovalResult(approved=False, feedback=feedback)


class LLMAgent:
    """A simple LLM Agent Class.

    Attributes:
        llm (LLM): The backbone LLM.
        tools_registry (dict[str, Tool]): The tools the LLM agent can equip
            the LLM with, represented as a dict.
        templates (LLMAgentTemplates): Prompt templates for LLM Agent.
        logger (logging.Logger): LLMAgent logger.
        subagents_registry (dict[str, SubAgentSpec]): Subagent registry,
            keyed by name. Built from the constructor list. Added in
            Chapter 9.
        a2a_agents_registry (dict[str, A2AAgentSpec]): A2A peer registry,
            keyed by name. Built from the constructor list. Unlike
            `MCPToolProvider` (`mcp__{name}__{tool_name}`), no namespace
            prefix: A2A peers dispatch through one generic tool with
            `name` as an enum value, not as a standalone callable tool
            name, so there's no flat-tool-namespace collision to guard
            against. Mirrors `subagents_registry`'s plain-name keying.
            Added in Chapter 10.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        templates: LLMAgentTemplates = default_templates,
        # added in ch07
        memories: list[Memory] | None = None,
        # added in ch09
        subagents: "list[SubAgentSpec] | None" = None,
        # added in ch10
        a2a_agents: list[A2AAgentSpec] | None = None,
    ):
        """Initialize an LLMAgent.

        Args:
            llm (LLM): The backbone LLM of the LLM agent.
            tools (list[Tool], optional): The set of tools with which the
                LLM can be equipped. Defaults to None.
            templates (LLMAgentTemplates): Prompt templates for LLM Agent.
            memories (list[Memory] | None): Episodic memory backends
                to consult at task start and update at task end. Defaults
                to None (no memory). Added in Chapter 7.
            subagents (list[SubAgentSpec] | None): Subagents this
                coordinator can delegate to. Defaults to None (no
                subagents). Added in Chapter 9.
            a2a_agents (list[A2AAgentSpec] | None): A2A peer agents this
                coordinator can dispatch to. Defaults to None (no A2A
                peers). Added in Chapter 10.
        """
        self.llm = llm
        tools = tools or []
        # validate no duplications in tool names
        if len({t.name for t in tools}) < len(tools):
            raise LLMAgentError(
                "Provided tool list contains duplicate tool names.",
            )
        self.tools_registry = {t.name: t for t in tools}
        self.templates = templates
        self.logger = get_logger(self.__class__.__name__)
        # added in ch07
        self.memories = memories or []
        # added in ch09
        subagents = subagents or []
        if len({s.name for s in subagents}) < len(subagents):
            raise LLMAgentError(
                "Provided subagent list contains duplicate names.",
            )
        self.subagents_registry: dict[str, SubAgentSpec] = {
            s.name: s for s in subagents
        }
        # added in ch10
        a2a_agents = a2a_agents or []
        if len({s.name for s in a2a_agents}) < len(a2a_agents):
            raise LLMAgentError(
                "Provided a2a_agents list contains duplicate names.",
            )
        self.a2a_agents_registry: dict[str, A2AAgentSpec] = {
            s.name: s for s in a2a_agents
        }

    @property
    def tools(self) -> list[Tool]:
        """Return tools as list."""
        return list(self.tools_registry.values())

    def add_tool(self, tool: Tool) -> Self:
        """Add a tool to the agents tool set.

        NOTE: Supports fluent style for convenience.

        Args:
            tool (Tool): The tool to equip the LLM agent.

        """
        if tool.name in self.tools_registry:
            raise LLMAgentError(f"Tool with name {tool.name} already exists.")
        self.tools_registry[tool.name] = tool
        return self

    class TaskHandler(asyncio.Future[TaskResult]):
        """Handler for processing tasks.

        Attributes:
            llm_agent (LLMAgent): The LLM agent.
            task: The task to execute.
            rollout: The execution log of the task.
            step_counter: The number of TaskSteps executed.
            logger: TaskHandler logger.
            skills_registry (dict[str, Skill]): Skills discovered at the
                start of each run, keyed by name. Added in Chapter 6.
            _explicit_only_skills (set[str]): Skill names excluded from the
                model-visible catalog for this run. They remain loadable via
                ``run_with_skill()``. Added in Chapter 6.
            _use_skill_tool (UseSkillTool | None): Task-scoped skill
                activation tool. Set when skills are discovered; ``None``
                otherwise. Added in Chapter 6.
            _use_subagent_tool (UseSubAgentTool | None): Task-scoped
                subagent dispatch tool. Set when subagents are registered
                on the agent; ``None`` otherwise. Added in Chapter 9.
            _use_a2a_agent_tool (UseA2AAgentTool | None): Task-scoped A2A
                peer dispatch tool. Set when A2A peers are registered on
                the agent; ``None`` otherwise. Added in Chapter 10.
        """

        def __init__(
            self,
            llm_agent: "LLMAgent",
            task: Task,
            # added in ch06
            skills_scopes: list[SkillScope] | None = None,
            explicit_only_skills: set[str] | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            """Initialize a TaskHandler.

            Args:
                llm_agent (LLMAgent): The LLM agent.
                task (Task): The task to process.
                skills_scopes (list[SkillScope] | None): Scopes to scan for
                    skills. Defaults to ``[USER, PROJECT]``. Added in
                    Chapter 6.
                explicit_only_skills (set[str] | None): Skill names to
                    exclude from the model catalog. Defaults to None.
                    Added in Chapter 6.
                *args: Additional positional arguments.
                **kwargs: Additional keyword arguments.
            """
            super().__init__(*args, **kwargs)
            self.llm_agent = llm_agent
            self.task = task
            self.rollout = ""
            self.step_counter = 0
            self._background_task: asyncio.Task | None = None
            self.logger = get_logger(self.__class__.__name__)
            # added in ch06
            _scopes = (
                skills_scopes
                if skills_scopes is not None
                else [SkillScope.USER, SkillScope.PROJECT]
            )
            self.skills_registry: dict[str, Skill] = discover_skills(_scopes)
            self._explicit_only_skills: set[str] = explicit_only_skills or set()
            self._use_skill_tool: UseSkillTool | None = (
                UseSkillTool(
                    skills_registry=self.skills_registry,
                    explicit_only_skills=self._explicit_only_skills,
                )
                if self.skills_registry
                else None
            )
            # added in ch07
            self._recalled_memories: str = ""
            # added in ch09
            self._use_subagent_tool: "UseSubAgentTool | None"
            if self.llm_agent.subagents_registry:
                from llm_agents_from_scratch.subagents.tools import (  # noqa: PLC0415
                    UseSubAgentTool,
                )

                self._use_subagent_tool = UseSubAgentTool(
                    subagents_registry=self.llm_agent.subagents_registry,
                )
            else:
                self._use_subagent_tool = None
            # added in ch10
            self._use_a2a_agent_tool: UseA2AAgentTool | None = (
                UseA2AAgentTool(
                    a2a_agents_registry=self.llm_agent.a2a_agents_registry,
                )
                if self.llm_agent.a2a_agents_registry
                else None
            )

        @property
        def background_task(self) -> asyncio.Task:
            """Get the background ~asyncio.Task for the handler."""
            if not self._background_task:
                raise TaskHandlerError(
                    "No background task is running for this handler.",
                )
            return self._background_task

        @background_task.setter
        def background_task(self, asyncio_task: asyncio.Task) -> None:
            """Setter for background_task."""
            if self._background_task is not None:
                raise TaskHandlerError(
                    "A background task has already been set.",
                )
            self._background_task = asyncio_task

        @property
        def _skills_catalog(self) -> str:
            """Return formatted skills catalog, or empty string.

            Added in Chapter 6.

            Builds the ``<available_skills>`` XML block from discovered
            skills that have not opted out of model-driven activation
            (i.e. ``disable-model-invocation`` is not set). Returns an
            empty string when no visible skills remain so callers can
            append it unconditionally without adding noise.
            """
            visible = [
                skill
                for name, skill in self.skills_registry.items()
                if name not in self._explicit_only_skills
            ]
            if not visible:
                return ""
            entries = "\n".join(skill.catalog() for skill in visible)
            return self.llm_agent.templates["skills_catalog"].format(
                skills=entries,
            )

        @property
        def _subagents_catalog(self) -> str:
            """Return formatted subagents catalog, or empty string.

            Added in Chapter 9.

            Builds the ``<available_subagents>`` XML block from subagents
            registered on the coordinator. Returns an empty string when no
            subagents are configured so callers can append it
            unconditionally without adding noise.
            """
            specs = self.llm_agent.subagents_registry.values()
            if not specs:
                return ""
            entries = "\n".join(s.catalog() for s in specs)
            return self.llm_agent.templates["subagents_catalog"].format(
                subagents=entries,
            )

        @property
        def _a2a_agents_catalog(self) -> str:
            """Return formatted A2A agents catalog, or empty string.

            Added in Chapter 10.

            Builds the ``<available_a2a_agents>`` XML block from A2A peers
            registered on the coordinator. Returns an empty string when no
            peers are configured so callers can append it unconditionally
            without adding noise.
            """
            specs = self.llm_agent.a2a_agents_registry.values()
            if not specs:
                return ""
            entries = "\n".join(s.catalog() for s in specs)
            return self.llm_agent.templates["a2a_agents_catalog"].format(
                a2a_agents=entries,
            )

        def _format_step_for_rollout(
            self,
            chat_history: list[ChatMessage],
        ) -> str:
            """Format a run_step's chat history as a rollout entry."""
            rollout_lines = ["=== Task Step Start ==="]
            for msg in chat_history:
                # don't include system messages in rollout
                content = msg.content
                role = msg.role

                if role == "system":
                    continue

                if role == "user":
                    # From the LLMAgent to the backbone LLM, but in a rollout
                    # we'll simplify to just LLM agent having a monologue
                    role = ChatRole.ASSISTANT
                    content = self.llm_agent.templates[
                        "step_rollout_content_instruction"
                    ].format(
                        instruction=content,
                    )

                if msg.tool_calls and msg.role == "assistant":
                    called_tools = "\n\n".join(
                        [
                            f"{t.model_dump_json(indent=4)}"
                            for t in msg.tool_calls
                        ],
                    )
                    content = self.llm_agent.templates[
                        "step_rollout_content_tool_call_request"
                    ].format(
                        called_tools=called_tools,
                    )

                rollout_lines.append(
                    self.llm_agent.templates[
                        "step_rollout_chat_message"
                    ].format(
                        actor=("🔧 " if role == ChatRole.TOOL else "💬 ")
                        + role.value,
                        content=content,
                    ),
                )

            rollout_lines.append(
                "=== Task Step End ===",
            )

            return "\n\n".join(rollout_lines)

        def _format_memories_for_system_prompt(
            self,
            memories: list[str],
        ) -> str:
            if memories:
                entries = "\n".join(memories)
                return self.llm_agent.templates["memories"].format(
                    memories=entries,
                )
            return ""

        async def get_next_step(
            self,
            previous_step_result: (
                TaskStepResult
                | RejectedTaskResult  # added in ch08
                | None
            ),
        ) -> TaskStep | TaskResult:
            """Based on previous step result, get next step or conclude task.

            Returns:
                TaskStep | TaskResult: Either the next step or the result of
                    the task.
            """
            if not previous_step_result:
                return TaskStep(
                    task_id=self.task.id_,
                    instruction=self.task.instruction,
                )
            # added in ch08: rejection bypasses LLM routing
            if isinstance(previous_step_result, RejectedTaskResult):
                self.logger.info(
                    f"🧠 New Step (rejection): {previous_step_result.feedback}",
                )
                return TaskStep(
                    task_id=self.task.id_,
                    instruction=self.llm_agent.templates[
                        "approval_rejection_feedback"
                    ].format(
                        content=previous_step_result.failed_result_content,
                        feedback=previous_step_result.feedback,
                    ),
                )
            self.logger.debug(f"🧵 Rollout: {self.rollout}")

            prompt = self.llm_agent.templates["get_next_step"].format(
                instruction=self.task.instruction,
                current_rollout=self.rollout,
                current_response=previous_step_result.content,
            )
            self.logger.debug(f"---NEXT STEP PROMPT: {prompt}")
            try:
                next_step = await self.llm_agent.llm.structured_output(
                    prompt=prompt,
                    mdl=NextStepDecision,
                )
                self.logger.debug(
                    f"---NEXT STEP: {next_step.model_dump_json()}",
                )
            except Exception as e:
                raise TaskHandlerError(
                    f"Failed to get next step: {str(e)}",
                ) from e

            if next_step.kind == "final_result":
                self.logger.info("No new step required.")
                retval = TaskResult(
                    task_id=self.task.id_,
                    content=previous_step_result.content,
                )
            else:  # next_step.kind == "next_step":
                self.logger.info(f"🧠 New Step: {next_step.content}")
                retval = TaskStep(
                    task_id=self.task.id_,
                    instruction=next_step.content,
                )

            return retval

        async def run_step(self, step: TaskStep) -> TaskStepResult:  # noqa: PLR0912, PLR0915
            """Run next step of a given task.

            A single step is executed through a single-turn conversation that
            the LLM agent has with itself. In other words, it is both the `user`
            providing the instruction (from `get_next_step`) as well as the
            `assistant` that provides the result.

            Args:
                step (TaskStep): The step to execute.

            Returns:
                TaskStepResult: The result of the step execution.
            """
            self.step_counter += 1
            self.logger.info(f"⚙️ Processing Step: {step.instruction}")
            self.logger.debug(f"🧵 Rollout: {self.rollout}")

            # include rollout as context in the system message
            system_message = ChatMessage(
                role=ChatRole.SYSTEM,
                content=self.llm_agent.templates[
                    "run_step_system_message"
                ].format(
                    llm_agent_system_message=self.llm_agent.templates[
                        "system_message"
                    ],
                    current_rollout=self.rollout,
                )
                if self.rollout
                else self.llm_agent.templates[
                    "run_step_system_message_without_rollout"
                ].format(
                    llm_agent_system_message=self.llm_agent.templates[
                        "system_message"
                    ],
                ),
            )

            # added in ch07: inject recalled memories
            if memories := self._recalled_memories:
                system_message = ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=f"{system_message.content}\n\n{memories}",
                )

            # added in ch06: bolt on skills catalog when skills are available
            if catalog := self._skills_catalog:
                system_message = ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=f"{system_message.content}\n\n{catalog}",
                )

            # added in ch09: bolt on subagents catalog when registered
            if catalog := self._subagents_catalog:
                system_message = ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=f"{system_message.content}\n\n{catalog}",
                )

            # added in ch10: bolt on A2A agents catalog when registered
            if catalog := self._a2a_agents_catalog:
                system_message = ChatMessage(
                    role=ChatRole.SYSTEM,
                    content=f"{system_message.content}\n\n{catalog}",
                )

            self.logger.debug(f"💬 SYSTEM: {system_message.content}")

            # fictitious user's input
            user_input = self.llm_agent.templates[
                "run_step_user_message"
            ].format(
                instruction=step.instruction,
            )
            self.logger.debug(f"💬 USER INPUT: {user_input}")

            # start single-turn conversation
            # added in ch06: include use_skill tool when skills are available
            # added in ch09: include use_subagent tool when registered
            # added in ch10: include use_a2a_agent tool when registered
            all_tools = (
                self.llm_agent.tools
                + ([self._use_skill_tool] if self._use_skill_tool else [])
                + ([self._use_subagent_tool] if self._use_subagent_tool else [])
                + (
                    [self._use_a2a_agent_tool]
                    if self._use_a2a_agent_tool
                    else []
                )
            )
            user_message, response_message = await self.llm_agent.llm.chat(
                input=user_input,
                chat_history=[system_message],
                tools=all_tools,
            )
            self.logger.debug(f"💬 ASSISTANT: {response_message.content}")

            # check if there are tool calls
            if response_message.tool_calls:

                async def _execute_tool_call(
                    tool_call: ToolCall,
                ) -> ToolCallResult:
                    self.logger.info(
                        f"🛠️ Executing Tool Call: {tool_call.tool_name}",
                    )
                    if tool := (
                        self.llm_agent.tools_registry.get(
                            tool_call.tool_name,
                        )
                        or (
                            self._use_skill_tool
                            if self._use_skill_tool
                            and tool_call.tool_name == self._use_skill_tool.name
                            else None
                        )
                        or (
                            self._use_subagent_tool
                            if self._use_subagent_tool
                            and tool_call.tool_name
                            == self._use_subagent_tool.name
                            else None
                        )
                        or (
                            self._use_a2a_agent_tool
                            if self._use_a2a_agent_tool
                            and tool_call.tool_name
                            == self._use_a2a_agent_tool.name
                            else None
                        )
                    ):
                        try:
                            if isinstance(tool, AsyncBaseTool):
                                tool_call_result = await tool(
                                    tool_call=tool_call,
                                )
                            else:
                                # run sync tools in a thread so the event loop
                                # stays free for concurrent async tool calls
                                tool_call_result = await asyncio.to_thread(
                                    tool,
                                    tool_call=tool_call,
                                )
                        except Exception as e:
                            error_details = {
                                "error_type": e.__class__.__name__,
                                "message": (
                                    f"Internal error while executing "
                                    f"tool: {e!s}"
                                ),
                            }
                            tool_call_result = ToolCallResult(
                                tool_call_id=tool_call.id_,
                                error=True,
                                content=json.dumps(error_details),
                            )
                        if tool_call_result.error:
                            self.logger.info(
                                "❌ Tool Call Failure: "
                                f"{tool_call_result.content}",
                            )
                        else:
                            self.logger.info(
                                "✅ Successful Tool Call: "
                                f"{tool_call_result.content}",
                            )
                    else:
                        error_msg = (
                            f"Tool with name {tool_call.tool_name} "
                            "doesn't exist."
                        )
                        tool_call_result = ToolCallResult(
                            tool_call_id=tool_call.id_,
                            error=True,
                            content=error_msg,
                        )
                        self.logger.info(
                            f"❌ Tool Call Failure: {tool_call_result.content}",
                        )
                    return tool_call_result

                tool_call_results = await asyncio.gather(
                    *[
                        _execute_tool_call(tc)
                        for tc in response_message.tool_calls
                    ],
                )

                # send tool call results back to llm to get result
                (
                    tool_messages,
                    another_response_message,
                ) = await self.llm_agent.llm.continue_chat_with_tool_results(  # noqa: E501
                    tool_call_results=tool_call_results,
                    chat_history=[
                        system_message,
                        user_message,
                        response_message,
                    ],
                )

                # get final content and update chat history
                if another_response_message.tool_calls:
                    # if has tool calls, we'll make them in the next step
                    final_content = "I need to make the following tool-calls:\n"
                    final_content += "\n".join(
                        t.model_dump_json(indent=4)
                        for t in another_response_message.tool_calls
                    )
                else:
                    final_content = another_response_message.content
                chat_history = (
                    [
                        system_message,
                        user_message,
                        response_message,
                    ]
                    + tool_messages
                    + [another_response_message]
                )
            else:
                final_content = response_message.content
                chat_history = [
                    system_message,
                    user_message,
                    response_message,
                ]

            # augment rollout from this turn
            formatted_step = self._format_step_for_rollout(
                chat_history=chat_history,
            )
            if self.rollout:
                self.rollout += "\n\n" + formatted_step

            else:
                self.rollout = formatted_step

            self.logger.info(
                f"✅ Step Result: {final_content}",
            )
            return TaskStepResult(
                task_step_id=step.id_,
                content=final_content,
            )

        async def load_memories(self) -> None:
            """Recall relevant episodes from all configured memory backends.

            Added in Chapter 7.

            Calls ``recall`` on each memory in ``self.llm_agent.memories``
            and stores the formatted string in ``self._recalled_memories``
            for prompt injection during ``run_step``. No-op when no memories
            are configured.
            """
            loaded = []
            for memory in self.llm_agent.memories:
                block = await memory.recall(self.task)
                loaded.append(block)
            self._recalled_memories = self._format_memories_for_system_prompt(
                loaded,
            )

        async def record_memory(
            self,
            result: TaskResult | None = None,
            error: Exception | None = None,
        ) -> None:
            """Build an Episode and write it to all configured memories.

            Exactly one of ``result`` or ``error`` must be provided.
            Called before ``set_result()`` / ``set_exception()`` so that
            ``await agent.run(task)`` returns only after the episode is
            written.

            Added in Chapter 7.

            Args:
                result (TaskResult | None): The successful task result.
                error (Exception | None): The exception from a failed task.

            Raises:
                RecordMemoryError: If neither ``result`` nor ``error`` is
                    provided.
            """
            if result is None and error is None:
                raise RecordMemoryError(
                    "record_memory() requires either result or error.",
                )
            episode = Episode(
                task=self.task,
                rollout=self.rollout,
                result=result,
                error=error,
            )
            for memory in self.llm_agent.memories:
                await memory.record(episode)

        async def request_approval(
            self,
            result: TaskResult,
        ) -> ApprovalResult:
            """Ask a human to approve or reject the proposed task result.

            Added in Chapter 8.

            Operator-gated human-in-the-loop pattern; unlike
            ``HumanInputTool``, the pause is not agent-initiated.
            Runs the blocking rich prompts in a thread via
            ``asyncio.to_thread``. Auto-approves on ``EOFError`` or
            ``KeyboardInterrupt`` (headless / interrupted terminal).

            Args:
                result (TaskResult): The proposed task result to review.

            Returns:
                ApprovalResult: The approval decision.
            """
            try:
                return await asyncio.to_thread(
                    _prompt_for_approval,
                    result,
                )
            except EOFError:
                self.logger.info(
                    "Approval prompt got EOF (headless); auto-approving.",
                )
                return ApprovalResult(approved=True, feedback="")
            except KeyboardInterrupt:
                self.logger.info(
                    "Approval prompt interrupted by operator; auto-approving.",
                )
                return ApprovalResult(
                    approved=True,
                    feedback="",
                )

    class SupervisedTaskHandler(TaskHandler):
        """TaskHandler for human-driven stepwise execution.

        Added in Chapter 8. Caller-driven human-in-the-loop pattern;
        unlike ``HumanInputTool`` (agent-initiated) and
        ``request_approval`` (operator-gated at result time), the human
        controls the entire execution cadence. Returned by
        ``run_supervised()``; the caller drives the loop manually via
        ``get_next_step()`` and ``run_step()`` and finalises execution
        with ``complete()`` or ``abort()``.
        """

        @property
        def background_task(self) -> asyncio.Task:
            """Not available in supervised mode."""
            raise TaskHandlerError(
                "SupervisedTaskHandler has no background task — "
                "execution is caller-driven via get_next_step() "
                "and run_step().",
            )

        @background_task.setter
        def background_task(self, asyncio_task: asyncio.Task) -> None:
            """Not available in supervised mode."""
            raise TaskHandlerError(
                "SupervisedTaskHandler has no background task — "
                "execution is caller-driven via get_next_step() "
                "and run_step().",
            )

        async def complete(self, result: TaskResult) -> None:
            """Accept the final result and resolve the handler.

            Added in Chapter 8.

            Args:
                result: The ``TaskResult`` to accept.
            """
            if not isinstance(result, TaskResult):
                raise TaskHandlerError(
                    f"complete() requires a TaskResult, "
                    f"got {type(result).__name__}.",
                )
            await self.record_memory(result=result)
            self.set_result(result)

        def reject(
            self,
            result: TaskResult,
            feedback: str,
        ) -> RejectedTaskResult:
            """Reject a proposed TaskResult and return feedback for re-routing.

            Added in Chapter 8.

            Args:
                result: The ``TaskResult`` to reject.
                feedback: Correction rationale passed back to the agent.

            Returns:
                RejectedTaskResult: Pass to ``get_next_step()`` to
                    re-enter the loop without consulting the LLM.
            """
            return RejectedTaskResult(
                failed_result_content=result.content,
                feedback=feedback,
            )

        async def abort(self, error: Exception | None = None) -> None:
            """Abort the supervised task and resolve the handler.

            Added in Chapter 8.

            Args:
                error: Exception to set. Defaults to
                    ``TaskHandlerError("Task aborted.")``.
            """
            err = error or TaskHandlerError("Task aborted.")
            await self.record_memory(error=err)
            self.set_exception(err)

    def run(
        self,
        task: Task,
        max_steps: int | None = None,
        # added in ch06
        skills_scopes: list[SkillScope] | None = None,
        explicit_only_skills: set[str] | None = None,
        # added in ch08
        with_approval: bool = False,
    ) -> TaskHandler:
        """Agent's processing loop for executing tasks.

        Args:
            task (Task): the Task to perform.
            max_steps (int | None): Maximum number of steps to run for task.
                Defaults to None.
            skills_scopes (list[SkillScope] | None): Scopes to scan for
                skills, in processing order (last wins on name collision).
                Defaults to ``[USER, PROJECT]``. Added in Chapter 6.
            explicit_only_skills (set[str] | None): Skill names to exclude
                from the model catalog for this run. They remain activatable
                via ``run_with_skill()``. Defaults to None. Added in
                Chapter 6.
            with_approval (bool): When ``True``, an end-of-loop human
                approval gate fires before each ``TaskResult`` is accepted.
                The human may approve (result is recorded and returned) or
                reject with feedback (feedback re-enters the loop as a new
                step). Rejections do not consume the step budget; pair with
                ``max_steps`` to bound repeated-rejection loops. Defaults
                to ``False``. Added in Chapter 8.

        Returns:
            TaskHandler: the TaskHandler object responsible for task execution.
        """
        task_handler = self.TaskHandler(
            llm_agent=self,
            task=task,
            skills_scopes=skills_scopes,
            explicit_only_skills=explicit_only_skills,
        )

        async def _process_loop() -> None:
            """The processing loop for the task handler execute its task.

            Cycle between get_next_step and run_step, until the task_handler
            is marked as done, either through a set result or an exception being
            set.
            """
            self.logger.info(f"🚀 Starting task: {task.instruction}")
            step_result = None

            # added in ch07
            await task_handler.load_memories()

            while not task_handler.done():
                try:
                    if task_handler.step_counter == max_steps:
                        raise MaxStepsReachedError("Max steps reached.")

                    next_step = await task_handler.get_next_step(step_result)

                    match next_step:
                        case TaskStep():
                            step_result = await task_handler.run_step(
                                next_step,
                            )
                        case TaskResult():
                            # added in ch08
                            if with_approval:
                                approval = await task_handler.request_approval(
                                    next_step,
                                )
                                if not approval.approved:
                                    step_result = RejectedTaskResult(
                                        failed_result_content=next_step.content,
                                        feedback=approval.feedback,
                                    )
                                    self.logger.info(
                                        "🔁 Task result rejected; "
                                        "re-entering loop with feedback.",
                                    )
                                    continue
                            await task_handler.record_memory(
                                result=next_step,
                            )  # added in ch07
                            task_handler.set_result(next_step)
                            self.logger.info(
                                f"🏁 Task completed: {next_step.content}",
                            )

                except Exception as e:
                    await task_handler.record_memory(error=e)  # added in ch07
                    task_handler.set_exception(e)

        task_handler.background_task = asyncio.create_task(_process_loop())

        return task_handler

    def run_with_skill(
        self,
        skill_name: str,
        prompt: str | None = None,
        max_steps: int | None = None,
        # added in ch08
        with_approval: bool = False,
    ) -> TaskHandler:
        """User-explicit skill activation: the programmatic slash command.

        Added in Chapter 6.

        Frames the task instruction to direct the model to activate the named
        skill as its first action, then runs the full agent loop. Relies on
        the model's tool-use ability to call ``use_skill`` — a fair assumption
        given the whole system depends on it. Unknown skill names are caught
        by the guard in ``UseSkillTool.__call__``.

        Args:
            skill_name (str): Name of the skill to activate.
            prompt (str | None): Optional instruction to pass alongside the
                skill activation. Defaults to None.
            max_steps (int | None): Maximum number of steps to run.
                Defaults to None.
            with_approval (bool): Passed through to ``run()``. Added in
                Chapter 8.

        Returns:
            TaskHandler: The handler responsible for task execution.
        """
        if prompt:
            instruction = EXPLICIT_SKILL_ACTIVATION_WITH_PROMPT_TEMPLATE.format(
                name=skill_name,
                prompt=prompt,
            )
        else:
            instruction = EXPLICIT_SKILL_ACTIVATION_TEMPLATE.format(
                name=skill_name,
            )
        task = Task(instruction=instruction)

        return self.run(
            task=task,
            max_steps=max_steps,
            # added in ch08
            with_approval=with_approval,
        )

    async def run_supervised(
        self,
        task: Task,
        skills_scopes: list[SkillScope] | None = None,
        explicit_only_skills: set[str] | None = None,
    ) -> SupervisedTaskHandler:
        """Human-driven stepwise task execution.

        Added in Chapter 8. Creates and returns a
        ``SupervisedTaskHandler`` with memories loaded, without starting
        the autonomous ``_process_loop``. The caller drives execution
        cell-by-cell via ``get_next_step()`` and ``run_step()``, then
        finalises with ``complete()`` or ``abort()``.

        Contrasts with ``run()``: supervised = human controls cadence;
        autonomous = agent runs to completion.

        Args:
            task: The task to perform.
            skills_scopes (list[SkillScope] | None): Scopes to scan for
                skills. Defaults to ``[USER, PROJECT]``.
            explicit_only_skills (set[str] | None): Skill names to
                exclude from the model catalog. Defaults to None.

        Returns:
            SupervisedTaskHandler: Ready for stepwise execution.
        """
        task_handler = self.SupervisedTaskHandler(
            llm_agent=self,
            task=task,
            skills_scopes=skills_scopes,
            explicit_only_skills=explicit_only_skills,
        )
        await task_handler.load_memories()
        return task_handler

    async def run_supervised_with_skill(
        self,
        skill_name: str,
        prompt: str | None = None,
        skills_scopes: list[SkillScope] | None = None,
        explicit_only_skills: set[str] | None = None,
    ) -> SupervisedTaskHandler:
        """Human-driven stepwise execution with a pre-loaded skill.

        Added in Chapter 8. Combines ``run_with_skill()`` (skill
        activation framing) with ``run_supervised()`` (caller-controlled
        cadence). The named skill is embedded in the task instruction so
        the model activates it as its first action; the caller then
        drives execution cell-by-cell via ``get_next_step()`` and
        ``run_step()``.

        Args:
            skill_name (str): Name of the skill to activate.
            prompt (str | None): Optional instruction to pass alongside
                the skill activation. Defaults to None.
            skills_scopes (list[SkillScope] | None): Scopes to scan for
                skills. Defaults to ``[USER, PROJECT]``.
            explicit_only_skills (set[str] | None): Skill names to
                exclude from the model catalog. Defaults to None.

        Returns:
            SupervisedTaskHandler: Ready for stepwise execution.
        """
        if prompt:
            instruction = EXPLICIT_SKILL_ACTIVATION_WITH_PROMPT_TEMPLATE.format(
                name=skill_name,
                prompt=prompt,
            )
        else:
            instruction = EXPLICIT_SKILL_ACTIVATION_TEMPLATE.format(
                name=skill_name,
            )
        task = Task(instruction=instruction)
        return await self.run_supervised(
            task=task,
            skills_scopes=skills_scopes,
            explicit_only_skills=explicit_only_skills,
        )
