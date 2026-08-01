"""SubAgentSpec — specification for a named sub-agent."""

from pydantic import BaseModel, ConfigDict, Field

from llm_agents_from_scratch.agent import LLMAgent


class SubAgentSpec(BaseModel):
    """Specification for a named sub-agent in a multi-agent system.

    Each ``SubAgentSpec`` entry registers a fully-built ``LLMAgent`` under a
    human-readable name. The name serves as the registry key on the parent
    coordinator and as the enum value the coordinator's ``UseSubAgentTool``
    presents to the LLM for dispatch.

    Attributes:
        name: Unique registry key for this sub-agent. Appears as an enum
            value in ``UseSubAgentTool``'s dispatch schema — must be
            human-readable and stable.
        description: Short routing signal shown to the coordinator LLM in
            the ``<available_subagents>`` catalog. Should describe capability,
            not implementation.
        agent: The fully-built sub-agent. Held as a live object; Pydantic's
            ``arbitrary_types_allowed`` is required for this field.
        max_steps: Optional cap on the number of steps the sub-agent may
            take per dispatch. Passed to ``agent.run()`` to bound runaway
            executions. ``None`` uses the agent's own default.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        description=(
            "Unique registry key for this sub-agent. Appears as an enum "
            "value in UseSubAgentTool's dispatch schema."
        ),
    )
    description: str = Field(
        description=(
            "Routing signal shown to the coordinator LLM in the "
            "<available_subagents> catalog."
        ),
    )
    agent: LLMAgent = Field(
        description="The fully-built sub-agent to dispatch tasks to.",
    )
    max_steps: int | None = Field(
        default=None,
        description=(
            "Optional cap on sub-agent steps per dispatch. "
            "None uses the agent's own default."
        ),
    )
