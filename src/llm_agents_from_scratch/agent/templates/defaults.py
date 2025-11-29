"""Default templates."""

DEFAULT_SYSTEM_MESSAGE = """You are a helpful assistant.

IMPORTANT: Do not include raw tool-call JSON in your responses. If you need to
use a tool, state your intent clearly (e.g., "I need to call the X tool with Y
parameters") and the system will execute it."""

DEFAULT_GET_NEXT_INSTRUCTION_PROMPT = """You are overseeing an assistant's
progress in accomplishing a user instruction. Provided below is the assistant's
current response to the original task instruction. Also provided is an
internal 'thinking' process of the assistant that the user has not seen.

Determine if the current response is sufficient to answer the original task
instruction.

IMPORTANT: If the assistant's response indicates they need to make a tool call
(e.g., "I need to call X tool..."), this is NOT a completed step. Generate a
next step instruction for them to execute that tool call.

In the case that the response is not sufficient, provide a new instruction to
the assistant to help them improve upon their current response.

<user-instruction>
{instruction}
</user-instruction>

<current-response>
{current_response}
</current-response>

<thinking-process>
{current_rollout}
</thinking-process>
"""

DEFAULT_RUN_STEP_USER_MESSAGE = "{instruction}"

DEFAULT_ROLLOUT_CONTRIBUTION_FROM_CHAT_MESSAGE = "💬 {role}: {content}"

DEFAULT_ROLLOUT_CONTRIBUTION_CONTENT_INSTRUCTION = (
    "The current instruction is '{instruction}'"
)

DEFAULT_ROLLOUT_CONTRIBUTION_CONTENT_TOOL_CALL_REQUEST = (
    "I need to make the following tool call(s):\n\n{called_tools}."
)

DEFAULT_RUN_STEP_SYSTEM_MESSAGE_WITHOUT_ROLLOUT = (
    """{llm_agent_system_message}"""
)

DEFAULT_RUN_STEP_SYSTEM_MESSAGE = """
{llm_agent_system_message}

Here is some past dialogue and context, where another assistant was working
towards completing the task.

<history>
{current_rollout}
</history>
""".strip()
