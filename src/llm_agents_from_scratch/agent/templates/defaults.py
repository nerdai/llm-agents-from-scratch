"""Default templates."""

DEFAULT_SYSTEM_MESSAGE = """You are a helpful assistant working through problems
step by step.

Think out loud as you work - reflect on what you observe, what it means, and
what to do next.

IMPORTANT: Do not include raw tool-call JSON in your responses. If you need to
use a tool, state your intent clearly (e.g., "I need to call the X tool with Y
parameters") and the system will execute it.

IMPORTANT: If past memories are available, use them. If a recalled <episode>
covers the same or a similar task, use its result directly rather than repeating
tool calls, unless explicitly instructed otherwise.
"""

DEFAULT_GET_NEXT_INSTRUCTION_PROMPT = """You are overseeing an assistant's
progress in accomplishing a user instruction.

<thinking-process>
{current_rollout}
</thinking-process>

<current-response>
{current_response}
</current-response>

<user-instruction>
{instruction}
</user-instruction>

DECISION CRITERIA:
- If current_response contains phrases like "I need to...", "Now I should...",
  "Next I will...", or describes a pending action → kind="next_step"
- If the task objective is FULLY achieved with a
  final answer → kind="final_result"
- When in doubt, choose "next_step"

CRITICAL: Statements like "I need to call monte_carlo_estimate" mean the task
is NOT complete. Generate a next_step instruction for the assistant to execute
that tool call.

NOTE: When kind="final_result", set content to an empty string. The
current_response will be used as the final answer.

What is your decision?
""".strip()

DEFAULT_RUN_STEP_USER_MESSAGE = "{instruction}"

DEFAULT_STEP_ROLLOUT_CHAT_MESSAGE = "{actor}: {content}"

DEFAULT_STEP_ROLLOUT_CONTENT_INSTRUCTION = (
    "My current instruction is '{instruction}'"
)

DEFAULT_STEP_ROLLOUT_CONTENT_TOOL_CALL_REQUEST = (
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

DEFAULT_SKILLS_CATALOG = """The following skills are available. To activate \
one, call the `from_scratch__use_skill` tool with the skill name.

<available_skills>
{skills}
</available_skills>""".strip()

DEFAULT_MEMORIES_BLOCK = """The following are memories from past episodes or
task executions.

IMPORTANT: If a recalled <episode> covers the same or a similar task, use its
result directly rather than repeating tool calls, unless explicitly instructed
otherwise.

<warning>
Review these carefully: apply any lessons or insights to
inform your approach, and if a past <episode> covers the same task, use its
result directly unless instructed otherwise.
</warning>

<memories>
{memories}
</memories>""".strip()

DEFAULT_APPROVAL_PROMPT = "Approve this result?"

DEFAULT_APPROVAL_RATIONALE_PROMPT = (
    "Provide your correction rationale for the agent to address"
)

DEFAULT_APPROVAL_REJECTION_FEEDBACK = """The human operator REJECTED your \
proposed task result. Revise your approach and try again.

<proposed-result>
{content}
</proposed-result>

<human-correction>
{feedback}
</human-correction>""".strip()
