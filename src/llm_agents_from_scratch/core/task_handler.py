"""Task Handler."""

import asyncio
from typing import Any

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.tool import AsyncBaseTool, BaseTool
from llm_agents_from_scratch.data_structures import (
    ChatMessage,
    ChatRole,
    Task,
    TaskStep,
    TaskStepResult,
    ToolCallResult,
)
from llm_agents_from_scratch.errors import TaskHandlerError

DEFAULT_GET_NEXT_INSTRUCTION_PROMPT = "{current_rollout}"
DEFAULT_SYSTEM_MESSAGE = "{original_instruction} {current_rollout}"
DEFAULT_USER_MESSAGE = "{instruction}"
DEFAULT_ROLLOUT_BLOCK_FROM_CHAT_MESSAGE = "{role}: {content}"


class TaskHandler(asyncio.Future):
    """Handler for processing tasks.

    Attributes:
        task: The task to execute.
        llm: The backbone LLM.
        tools_registry: The tools the LLM agent can use represented as a dict.
        rollout: The execution log of the task.
    """

    def __init__(
        self,
        task: Task,
        llm: BaseLLM,
        tools: list[BaseTool | AsyncBaseTool],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize a TaskHandler.

        Args:
            task (Task): The task to process.
            llm (BaseLLM): The backbone LLM.
            tools (list[BaseTool]): The tools the LLM can use.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.task = task
        self.llm = llm
        self.tools_registry = {t.name: t for t in tools}
        self.rollout = ""
        self._background_task: asyncio.Task | None = None
        self._lock: asyncio.Lock = asyncio.Lock()

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
            if msg.role == "system":
                continue
            rollout_contributions.append(
                DEFAULT_ROLLOUT_BLOCK_FROM_CHAT_MESSAGE.format(
                    role=msg.role.value,
                    content=msg.content,
                ),
            )
        return "\n".join(rollout_contributions)

    async def get_next_step(self) -> TaskStep:
        """Based on task progress, determine next step.

        Returns:
            TaskStep: The next step to run, if `None` then Task is done.
        """
        async with self._lock:
            rollout = self.rollout

        if rollout == "":
            return TaskStep(instruction=self.task.instruction, last_step=False)

        prompt = DEFAULT_GET_NEXT_INSTRUCTION_PROMPT.format(
            current_rollout=rollout,
        )
        try:
            task_step = await self.llm.structured_output(
                prompt=prompt,
                mdl=TaskStep,
            )
        except Exception as e:
            raise TaskHandlerError(f"Failed to get next step: {str(e)}") from e

        return task_step

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
        async with self._lock:
            rollout = self.rollout

        # include rollout as context in the system message
        system_message = ChatMessage(
            role=ChatRole.SYSTEM,
            content=DEFAULT_SYSTEM_MESSAGE.format(
                original_instruction=self.task.instruction,
                current_rollout=rollout,
            ),
        )
        user_message = ChatMessage(
            role=ChatRole.USER,
            content=DEFAULT_USER_MESSAGE.format(
                instruction=step.instruction,
            ),
        )

        # start conversation
        response = await self.llm.chat(
            input=user_message.content,
            chat_messages=[system_message],
            tools=list(self.tools_registry.values()),
        )

        chat_history = [
            system_message,
            user_message,
            response,
        ]

        # see if there are tool calls
        if response.tool_calls:
            tool_call_results = []
            for tool_call in response.tool_calls:
                if tool := self.tools_registry.get(tool_call.tool_name):
                    if isinstance(tool, AsyncBaseTool):
                        tool_call_result = await tool(tool_call=tool_call)
                    else:
                        tool_call_result = tool(tool_call=tool_call)
                else:
                    error_msg = (
                        f"Tool with name {tool_call.tool_name} doesn't exist.",
                    )
                    tool_call_result = ToolCallResult(
                        tool_call=tool_call,
                        error=True,
                        content=error_msg,
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

        return TaskStepResult(
            task_step=step,
            content=final_content,
        )
