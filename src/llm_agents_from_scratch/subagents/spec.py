"""SubAgentSpec — specification for a named sub-agent."""

from pydantic import BaseModel, ConfigDict, Field

from llm_agents_from_scratch.agent import LLMAgentBuilder
from llm_agents_from_scratch.data_structures.skill import SkillScope

from .constants import CATALOG_SPEC_TEMPLATE


class SubAgentSpec(BaseModel):
    """Specification for a named sub-agent in a multi-agent system.

    Each ``SubAgentSpec`` entry registers a build recipe for a sub-agent
    under a human-readable name. The name serves as the registry key on
    the parent coordinator and as the enum value the coordinator's
    ``UseSubAgentTool`` presents to the LLM for dispatch.

    The spec is pure data: it never constructs or holds a live agent.
    ``UseSubAgentTool`` builds a fresh ``LLMAgent`` from ``builder`` on
    every dispatch. Whatever the builder was given directly (an
    ``LLM``, ``Memory`` stores, ``MCPToolProvider`` instances) is reused
    as-is across builds — only the ``LLMAgent`` shell itself is rebuilt
    each time, mirroring the existing fresh-``TaskHandler``-per-``run()``
    pattern.

    Not in ``data_structures/``: that package is the zero-dependency
    bottom layer (stdlib/pydantic imports only, no framework
    cross-references), reserved for passive records like
    ``SkillFrontmatter``. ``builder`` holds a live ``LLMAgentBuilder``
    (why ``arbitrary_types_allowed=True`` is needed) and ``catalog()``
    is real behavior — this spec matches ``Skill``'s shape, not
    ``SkillFrontmatter``'s. Moving it would also create a real import
    cycle: ``LLMAgent`` only imports ``SubAgentSpec`` under
    ``TYPE_CHECKING`` today precisely to avoid
    ``agent.llm_agent`` → ``subagents.spec`` → ``agent.builder`` →
    ``agent.llm_agent``; ``data_structures/`` is imported eagerly
    everywhere, so that guard would stop working. See
    ``A2AAgentSpec`` for the same reasoning, spelled out in full.

    Attributes:
        name: Unique registry key for this sub-agent. Appears as an enum
            value in ``UseSubAgentTool``'s dispatch schema — must be
            human-readable and stable.
        description: Short routing signal shown to the coordinator LLM in
            the ``<available_subagents>`` catalog. Should describe capability,
            not implementation.
        builder: The recipe used to build this sub-agent fresh on every
            dispatch. Held as a live object; Pydantic's
            ``arbitrary_types_allowed`` is required for this field.
        max_steps: Optional cap on the number of steps the sub-agent may
            take per dispatch. Passed to ``agent.run()`` to bound runaway
            executions. ``None`` uses the agent's own default.
        skills_scopes: Optional scopes to scan for skills on this
            sub-agent's dispatches. Passed to ``agent.run()``. ``None``
            uses the agent's own default (``[USER, PROJECT]``).
        explicit_only_skills: Optional skill names to exclude from this
            sub-agent's model catalog on dispatch. Passed to
            ``agent.run()``. ``None`` uses the agent's own default (no
            exclusions).
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
    builder: LLMAgentBuilder = Field(
        description="The recipe used to build this sub-agent per dispatch.",
    )
    max_steps: int | None = Field(
        default=None,
        description=(
            "Optional cap on sub-agent steps per dispatch. "
            "None uses the agent's own default."
        ),
    )
    skills_scopes: list[SkillScope] | None = Field(
        default=None,
        description=(
            "Optional scopes to scan for skills on this sub-agent's "
            "dispatches. None uses the agent's own default."
        ),
    )
    explicit_only_skills: set[str] | None = Field(
        default=None,
        description=(
            "Optional skill names to exclude from this sub-agent's model "
            "catalog on dispatch. None uses the agent's own default."
        ),
    )

    def catalog(self) -> str:
        """Return XML entry for this spec in the sub-agents catalog."""
        return CATALOG_SPEC_TEMPLATE.format(
            name=self.name,
            description=self.description,
        )
