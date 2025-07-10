"""Task Handler."""

import asyncio
from typing import Any

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import AsyncBaseTool, BaseTool
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ChatRole,
    GetNextStep,
    Task,
    TaskResult,
    TaskStep,
    TaskStepResult,
    ToolCallResult,
)
from llm_agents_from_scratch.errors import TaskHandlerError
from llm_agents_from_scratch.logger import get_logger

from .templates import TaskHandlerTemplates, default_task_handler_templates


class TaskHandler(asyncio.Future):
    """Handler for processing tasks.

    Attributes:
        task: The task to execute.
        llm: The backbone LLM.
        tools_registry: The tools the LLM agent can use represented as a dict.
        templates: Associated prompt templates.
        rollout: The execution log of the task.
        logger: TaskHandler logger.
    """

    def __init__(
        self,
        task: Task,
        llm: BaseLLM,
        tools: list[BaseTool | AsyncBaseTool],
        templates: TaskHandlerTemplates = default_task_handler_templates,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize a TaskHandler.

        Args:
            task (Task): The task to process.
            llm (BaseLLM): The backbone LLM.
            tools (list[BaseTool]): The tools the LLM can use.
            templates (TaskHandlerTemplates): Associated prompt templates.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.task = task
        self.llm = llm
        self.tools_registry = {t.name: t for t in tools}
        self.rollout = ""
        self.templates = templates
        self._background_task: asyncio.Task | None = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self.logger = get_logger(self.__class__.__name__)

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
            raise TaskHandlerError("A background task has already been set.")
        self._background_task = asyncio_task

    def _rollout_contribution_from_single_run_step(
        self,
        chat_history: list[ChatMessage],
    ) -> str:
        """Update rollout after a run_step execution."""
        rollout_contributions = []
        for msg in chat_history:
            # don't include system messages in rollout
            content = msg.content
            role = msg.role

            if role == "system":
                continue

            if role == "user":
                role = ChatRole.ASSISTANT

            if msg.tool_calls and msg.role == "assistant":
                content = (
                    "I need to make a tool call(s) to "
                    f"{', '.join([t.tool_name for t in msg.tool_calls])}"
                )

            rollout_contributions.append(
                self.templates["rollout_block_from_chat_message"].format(
                    role=role.value,
                    content=content,
                ),
            )
        return "\n".join(rollout_contributions)

    async def get_next_step(
        self,
        previous_step_result: TaskStepResult | None,
    ) -> TaskStep | TaskResult:
        """Based on most previous step result, get next step or conclude task.

        Returns:
            TaskStep | TaskResult: Either the next step or the result of the
                task.
        """
        async with self._lock:
            rollout = self.rollout
            self.logger.debug(f"🧵 Rollout: {rollout}")

        if not previous_step_result:
            return TaskStep(
                instruction=self.task.instruction,
                last_step=False,
            )
        prompt = self.templates["get_next_step"].format(
            instruction=self.task.instruction,
            current_rollout=rollout,
            current_response=previous_step_result.content,
        )
        self.logger.debug(f"---NEXT STEP PROMPT: {prompt}")
        try:
            next_step = await self.llm.structured_output(
                prompt=prompt,
                mdl=GetNextStep,
            )
            self.logger.debug(f"---NEXT STEP: {next_step.model_dump_json()}")
        except Exception as e:
            raise TaskHandlerError(
                f"Failed to get next step: {str(e)}",
            ) from e

        task_step = next_step.task_step
        task_result = next_step.task_result

        if task_result:
            self.logger.info("No new step required.")
            return task_result

        if task_step:
            self.logger.info(f"🧠 New Step: {task_step.instruction}")
            return task_step

        error_msg = (
            "Getting next step failed. Structured output didn't yield a "
            "`TaskResult` nor a `TaskStep`."
        )
        raise TaskHandlerError(error_msg)

    async def run_step(self, step: TaskStep) -> TaskStepResult:
        """Run next step of a given task.

        A single step is executed through a single-turn conversation that the
        LLM agent has with itself. In other words, it is both the `user`
        providing the instruction (from `get_next_step`) as well as the
        `assistant` that provides the result.

        Args:
            step (TaskStep): The step to execute.

        Returns:
            TaskStepResult: The result of the step execution.
        """
        self.logger.info(f"⚙️ Processing Step: {step.instruction}")
        async with self._lock:
            rollout = self.rollout
            self.logger.debug(f"🧵 Rollout: {rollout}")

        # include rollout as context in the system message
        system_message = ChatMessage(
            role=ChatRole.SYSTEM,
            content=self.templates["system_message"].format(
                original_instruction=self.task.instruction,
                current_rollout=rollout,
            )
            if rollout
            else self.templates["system_message_without_rollout"],
        )
        self.logger.debug(f"💬 SYSTEM: {system_message.content}")
        user_message = ChatMessage(
            role=ChatRole.USER,
            content=self.templates["user_message"].format(
                instruction=step.instruction,
            ),
        )
        self.logger.debug(f"💬 USER: {user_message.content}")

        # start conversation
        response = await self.llm.chat(
            input=user_message.content,
            chat_messages=[system_message],
            tools=list(self.tools_registry.values()),
        )
        self.logger.debug(f"💬 ASSISTANT: {response.content}")

        chat_history = [
            system_message,
            user_message,
            response,
        ]

        # see if there are tool calls
        if response.tool_calls:
            tool_call_results = []
            for tool_call in response.tool_calls:
                self.logger.info(
                    f"🛠️ Executing Tool Call: {tool_call.tool_name}",
                )
                if tool := self.tools_registry.get(tool_call.tool_name):
                    if isinstance(tool, AsyncBaseTool):
                        tool_call_result = await tool(tool_call=tool_call)
                    else:
                        tool_call_result = tool(tool_call=tool_call)
                    self.logger.info(
                        f"✅ Successful Tool Call: {tool_call_result.content}",
                    )
                else:
                    error_msg = (
                        f"Tool with name {tool_call.tool_name} doesn't exist.",
                    )
                    tool_call_result = ToolCallResult(
                        tool_call=tool_call,
                        error=True,
                        content=error_msg,
                    )
                    self.logger.info(
                        f"❌ Tool Call Failure: {tool_call_result.content}",
                    )
                tool_call_results.append(tool_call_result)

            # send tool call results back to llm to get result
            new_messages = (
                await self.llm.continue_conversation_with_tool_results(
                    tool_call_results=tool_call_results,
                    chat_messages=chat_history,
                )
            )

            # get final content and update chat history
            final_content = new_messages[-1].content
            chat_history += new_messages
        else:
            final_content = response.content

        # augment rollout from this turn
        async with self._lock:
            self.rollout += self._rollout_contribution_from_single_run_step(
                chat_history=chat_history,
            )

        self.logger.info(
            f"✅ Step Result: {final_content}",
        )
        return TaskStepResult(
            task_step=step,
            content=final_content,
        )
