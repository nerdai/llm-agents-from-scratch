"""Prompt templates for LLMAgent (TaskHandler)."""

from typing import TypedDict

from .defaults import (
    DEFAULT_GET_NEXT_INSTRUCTION_PROMPT,
    DEFAULT_ROLLOUT_CONTRIBUTION_CONTENT_INSTRUCTION,
    DEFAULT_ROLLOUT_CONTRIBUTION_CONTENT_TOOL_CALL_REQUEST,
    DEFAULT_ROLLOUT_CONTRIBUTION_FROM_CHAT_MESSAGE,
    DEFAULT_RUN_STEP_SYSTEM_MESSAGE,
    DEFAULT_RUN_STEP_SYSTEM_MESSAGE_WITHOUT_ROLLOUT,
    DEFAULT_RUN_STEP_USER_MESSAGE,
    DEFAULT_SYSTEM_MESSAGE,
)


class LLMAgentTemplates(TypedDict):
    """Prompt templates dict for LLMAgent."""

    system_message: str
    # for task handler
    get_next_step: str
    rollout_contribution_from_chat_message: str
    rollout_contribution_content_instruction: str
    rollout_contribution_content_tool_call_request: str
    run_step_system_message_without_rollout: str
    run_step_system_message: str
    run_step_user_message: str


default_templates = LLMAgentTemplates(
    system_message=DEFAULT_SYSTEM_MESSAGE,
    get_next_step=DEFAULT_GET_NEXT_INSTRUCTION_PROMPT,
    rollout_contribution_from_chat_message=DEFAULT_ROLLOUT_CONTRIBUTION_FROM_CHAT_MESSAGE,
    rollout_contribution_content_instruction=DEFAULT_ROLLOUT_CONTRIBUTION_CONTENT_INSTRUCTION,
    rollout_contribution_content_tool_call_request=DEFAULT_ROLLOUT_CONTRIBUTION_CONTENT_TOOL_CALL_REQUEST,
    run_step_system_message_without_rollout=DEFAULT_RUN_STEP_SYSTEM_MESSAGE_WITHOUT_ROLLOUT,
    run_step_system_message=DEFAULT_RUN_STEP_SYSTEM_MESSAGE,
    run_step_user_message=DEFAULT_RUN_STEP_USER_MESSAGE,
)
