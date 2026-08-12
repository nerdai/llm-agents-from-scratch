"""SubAgentSpec — specification for a named sub-agent."""

from llm_agents_from_scratch.agent import LLMAgentBuilder
from llm_agents_from_scratch.data_structures.skill import SkillScope

from .constants import CATALOG_SPEC_TEMPLATE


class SubAgentSpec:
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

    A plain class, not a pydantic ``BaseModel``, for the same reason
    ``Skill``/``Memory``/``LLMAgentBuilder`` are plain classes: those
    are reserved for behavior-bearing objects that wrap a live
    collaborator, not passive serializable data. ``BaseModel`` would
    only have bought required-field validation (a plain ``__init__``
    already raises ``TypeError`` on missing required args) at the cost
    of needing ``arbitrary_types_allowed=True`` for ``builder``, a
    non-pydantic type.

    Attributes:
        name: Unique registry key for this sub-agent. Appears as an enum
            value in ``UseSubAgentTool``'s dispatch schema — must be
            human-readable and stable.
        description: Short routing signal shown to the coordinator LLM in
            the ``<available_subagents>`` catalog. Should describe capability,
            not implementation.
        builder: The recipe used to build this sub-agent fresh on every
            dispatch.
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

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        name: str,
        description: str,
        builder: LLMAgentBuilder,
        max_steps: int | None = None,
        skills_scopes: list[SkillScope] | None = None,
        explicit_only_skills: set[str] | None = None,
    ) -> None:
        """Initialise a SubAgentSpec.

        Args:
            name: Unique registry key for this sub-agent.
            description: Routing signal shown to the coordinator LLM in
                the ``<available_subagents>`` catalog.
            builder: The recipe used to build this sub-agent per
                dispatch.
            max_steps: Optional cap on sub-agent steps per dispatch.
                Defaults to ``None`` (agent's own default).
            skills_scopes: Optional scopes to scan for skills on this
                sub-agent's dispatches. Defaults to ``None`` (agent's
                own default).
            explicit_only_skills: Optional skill names to exclude from
                this sub-agent's model catalog on dispatch. Defaults to
                ``None`` (agent's own default).
        """
        self.name = name
        self.description = description
        self.builder = builder
        self.max_steps = max_steps
        self.skills_scopes = skills_scopes
        self.explicit_only_skills = explicit_only_skills

    def __repr__(self) -> str:
        """Readable repr, mirroring the fields pydantic would have shown."""
        return (
            f"{type(self).__name__}("
            f"name={self.name!r}, "
            f"description={self.description!r}, "
            f"builder={self.builder!r}, "
            f"max_steps={self.max_steps!r}, "
            f"skills_scopes={self.skills_scopes!r}, "
            f"explicit_only_skills={self.explicit_only_skills!r})"
        )

    def catalog(self) -> str:
        """Return XML entry for this spec in the sub-agents catalog."""
        return CATALOG_SPEC_TEMPLATE.format(
            name=self.name,
            description=self.description,
        )
