"""Default templates."""

DEFAULT_SYSTEM_MESSAGE = """You are a helpful assistant working through problems
step by step.

Think out loud as you work - reflect on what you observe, what it means, and
what to do next.

IMPORTANT: Do not include raw tool-call JSON in your responses. If you need to
use a tool, state your intent clearly (e.g., "I need to call the X tool with Y
parameters") and the system will execute it."""

DEFAULT_GET_NEXT_INSTRUCTION_PROMPT = """You are overseeing an assistant's
progress in accomplishing a user instruction. The assistant thinks out loud
as they work through the problem.

Provided below is the assistant's current response and their internal thinking.

Determine if the current response is sufficient to answer the original task
instruction.

IMPORTANT: If the assistant's response indicates they need to make a tool call
(e.g., "I need to call X tool..."), this is NOT a completed step. Do not
WAIT for that tool call, instead generate a next step instruction for them to
execute it.

If the response is not sufficient, provide a new instruction to help them
continue their reasoning.

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

DEFAULT_ROLLOUT_CONTRIBUTION_FROM_CHAT_MESSAGE = "{actor}: {content}"

DEFAULT_ROLLOUT_CONTRIBUTION_CONTENT_INSTRUCTION = (
    "My current instruction is '{instruction}'"
)

DEFAULT_ROLLOUT_CONTRIBUTION_CONTENT_TOOL_CALL_REQUEST = (
    "I need to make the following tool call(s):\n\n{called_tools}."
)

DEFAULT_RUN_STEP_SYSTEM_MESSAGE_WITHOUT_ROLLOUT = (
    """{llm_agent_system_message}"""
)

DEFAULT_RUN_STEP_SYSTEM_MESSAGE = """
{llm_agent_system_message}

You are in the middle of working through a task. Here's your thinking so far:

<my-thinking>
{current_rollout}
</my-thinking>

Continue your train of thought from where you left off.
""".strip()
