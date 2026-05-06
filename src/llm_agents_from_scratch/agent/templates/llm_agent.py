"""Prompt templates for LLMAgent (TaskHandler)."""

from typing import TypedDict

from .defaults import (
    DEFAULT_GET_NEXT_INSTRUCTION_PROMPT,
    DEFAULT_RUN_STEP_SYSTEM_MESSAGE,
    DEFAULT_RUN_STEP_SYSTEM_MESSAGE_WITHOUT_ROLLOUT,
    DEFAULT_RUN_STEP_USER_MESSAGE,
    DEFAULT_SKILLS_CATALOG,
    DEFAULT_STEP_ROLLOUT_CHAT_MESSAGE,
    DEFAULT_STEP_ROLLOUT_CONTENT_INSTRUCTION,
    DEFAULT_STEP_ROLLOUT_CONTENT_TOOL_CALL_REQUEST,
    DEFAULT_SYSTEM_MESSAGE,
)


class LLMAgentTemplates(TypedDict):
    """Prompt templates dict for LLMAgent."""

    system_message: str
    # for task handler
    get_next_step: str
    step_rollout_chat_message: str
    step_rollout_content_instruction: str
    step_rollout_content_tool_call_request: str
    run_step_system_message_without_rollout: str
    run_step_system_message: str
    run_step_user_message: str
    # added in ch06
    skills_catalog: str


default_templates = LLMAgentTemplates(
    system_message=DEFAULT_SYSTEM_MESSAGE,
    get_next_step=DEFAULT_GET_NEXT_INSTRUCTION_PROMPT,
    step_rollout_chat_message=DEFAULT_STEP_ROLLOUT_CHAT_MESSAGE,
    step_rollout_content_instruction=DEFAULT_STEP_ROLLOUT_CONTENT_INSTRUCTION,
    step_rollout_content_tool_call_request=DEFAULT_STEP_ROLLOUT_CONTENT_TOOL_CALL_REQUEST,
    run_step_system_message_without_rollout=DEFAULT_RUN_STEP_SYSTEM_MESSAGE_WITHOUT_ROLLOUT,
    run_step_system_message=DEFAULT_RUN_STEP_SYSTEM_MESSAGE,
    run_step_user_message=DEFAULT_RUN_STEP_USER_MESSAGE,
    # added in ch06
    skills_catalog=DEFAULT_SKILLS_CATALOG,
)
