"""Agent Module."""

import asyncio
from typing import Any

from typing_extensions import Self

from llm_agents_from_scratch.base.llm import LLM
from llm_agents_from_scratch.base.tool import AsyncBaseTool, Tool
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ChatRole,
    NextStepDecision,
    Task,
    TaskResult,
    TaskStep,
    TaskStepResult,
    ToolCallResult,
)
from llm_agents_from_scratch.data_structures.skill import SkillScope
from llm_agents_from_scratch.errors import (
    LLMAgentError,
    MaxStepsReachedError,
    TaskHandlerError,
)
from llm_agents_from_scratch.logger import get_logger
from llm_agents_from_scratch.skills.constants import (
    EXPLICIT_SKILL_ACTIVATION_TEMPLATE,
    EXPLICIT_SKILL_ACTIVATION_WITH_PROMPT_TEMPLATE,
)
from llm_agents_from_scratch.skills.discovery import discover_skills
from llm_agents_from_scratch.skills.skill import Skill
from llm_agents_from_scratch.skills.tools import UseSkillTool

from .templates import LLMAgentTemplates, default_templates


class LLMAgent:
    """A simple LLM Agent Class.

    Attributes:
        llm (LLM): The backbone LLM.
        tools_registry (dict[str, Tool]): The tools the LLM agent can equip
            the LLM with, represented as a dict.
        templates (LLMAgentTemplates): Prompt templates for LLM Agent.
        logger (logging.Logger): LLMAgent logger.
    """

    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        templates: LLMAgentTemplates = default_templates,
    ):
        """Initialize an LLMAgent.

        Args:
            llm (LLM): The backbone LLM of the LLM agent.
            tools (list[Tool], optional): The set of tools with which the
                LLM can be equipped. Defaults to None.
            templates (LLMAgentTemplates): Prompt templates for LLM Agent.
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

    class TaskHandler(asyncio.Future):
        """Handler for processing tasks.

        Attributes:
            llm_agent (LLMAgent): The LLM agent.
            task: The task to execute.
            rollout: The execution log of the task.
            step_counter: The number of TaskSteps executed.
            logger: TaskHandler logger.
            skills (dict[str, Skill]): Skills discovered at the start of
                each run, keyed by name. Added in Chapter 6.
            _explicit_only_skills (set[str]): Skill names excluded from the
                model-visible catalog for this run. They remain loadable via
                ``run_with_skill()``. Added in Chapter 6.
            _use_skill_tool (UseSkillTool | None): Task-scoped skill
                activation tool. Set when skills are discovered; ``None``
                otherwise. Added in Chapter 6.
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
            self.skills: dict[str, Skill] = discover_skills(_scopes)
            self._explicit_only_skills: set[str] = explicit_only_skills or set()
            self._use_skill_tool: UseSkillTool | None = (
                UseSkillTool(
                    skills=self.skills,
                    explicit_only_skills=self._explicit_only_skills,
                )
                if self.skills
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

            Builds the ``<available_skills>`` XML block from discovered
            skills that have not opted out of model-driven activation
            (i.e. ``disable-model-invocation`` is not set). Returns an
            empty string when no visible skills remain so callers can
            append it unconditionally without adding noise. Added in
            Chapter 6.
            """
            visible = [
                skill
                for name, skill in self.skills.items()
                if name not in self._explicit_only_skills
            ]
            if not visible:
                return ""
            entries = "\n".join(skill.catalog() for skill in visible)
            return self.llm_agent.templates["skills_catalog"].format(
                skills=entries,
            )

        def _format_step_for_rollout(
            self,
            chat_history: list[ChatMessage],
        ) -> str:
            """Format a run_step's chat history as a rollout entry."""
            rollout_contributions = ["=== Task Step Start ==="]
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

                rollout_contributions.append(
                    self.llm_agent.templates[
                        "step_rollout_chat_message"
                    ].format(
                        actor=("🔧 " if role == ChatRole.TOOL else "💬 ")
                        + role.value,
                        content=content,
                    ),
                )

            rollout_contributions.append(
                "=== Task Step End ===",
            )

            return "\n\n".join(rollout_contributions)

        async def get_next_step(
            self,
            previous_step_result: TaskStepResult | None,
        ) -> TaskStep | TaskResult:
            """Based on previous step result, get next step or conclude task.

            Returns:
                TaskStep | TaskResult: Either the next step or the result of the
                    task.
            """
            if not previous_step_result:
                return TaskStep(
                    task_id=self.task.id_,
                    instruction=self.task.instruction,
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

        async def run_step(self, step: TaskStep) -> TaskStepResult:
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
            # added in ch06: bolt on skills catalog when skills are available
            if catalog := self._skills_catalog:
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
            all_tools = self.llm_agent.tools + (
                [self._use_skill_tool] if self._use_skill_tool else []
            )
            user_message, response_message = await self.llm_agent.llm.chat(
                input=user_input,
                chat_history=[system_message],
                tools=all_tools,
            )
            self.logger.debug(f"💬 ASSISTANT: {response_message.content}")

            # check if there are tool calls
            if response_message.tool_calls:
                tool_call_results = []
                for tool_call in response_message.tool_calls:
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
                    ):
                        if isinstance(tool, AsyncBaseTool):
                            tool_call_result = await tool(tool_call=tool_call)
                        else:
                            tool_call_result = tool(tool_call=tool_call)
                        msg = (
                            "✅ Successful Tool Call: "
                            f"{tool_call_result.content}"
                        )
                        self.logger.info(msg)
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
                    tool_call_results.append(tool_call_result)

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
            rollout_contribution = self._format_step_for_rollout(
                chat_history=chat_history,
            )
            if self.rollout:
                self.rollout += "\n\n" + rollout_contribution

            else:
                self.rollout = rollout_contribution

            self.logger.info(
                f"✅ Step Result: {final_content}",
            )
            return TaskStepResult(
                task_step_id=step.id_,
                content=final_content,
            )

    def run(
        self,
        task: Task,
        max_steps: int | None = None,
        # added in ch06
        skills_scopes: list[SkillScope] | None = None,
        explicit_only_skills: set[str] | None = None,
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
                            task_handler.set_result(next_step)
                            self.logger.info(
                                f"🏁 Task completed: {next_step.content}",
                            )

                except Exception as e:
                    task_handler.set_exception(e)

        task_handler.background_task = asyncio.create_task(_process_loop())

        return task_handler

    def run_with_skill(
        self,
        skill_name: str,
        prompt: str | None = None,
        max_steps: int | None = None,
    ) -> TaskHandler:
        """User-explicit skill activation: the programmatic slash command.

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

        return self.run(task=task, max_steps=max_steps)
