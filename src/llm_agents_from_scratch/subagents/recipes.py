"""Standard subagent recipes for common default rosters.

Factories, not module-level constants: a ``SubAgentSpec`` wraps a live
``LLMAgent``, which needs an LLM — there is no zero-config answer the
way ``DEFAULT_TOOLS`` is for tools. Roster is ``general`` + ``explore``,
the intersection of the subagent rosters shipped by other harnesses
(a full-tools generalist and a read-only, cheap-to-run lookup agent).
"""

from llm_agents_from_scratch.agent import LLMAgent
from llm_agents_from_scratch.base.llm import LLM
from llm_agents_from_scratch.memory.memory import Memory
from llm_agents_from_scratch.tools.default import DEFAULT_TOOLS, ReadFileTool

from .constants import (
    EXPLORE_MAX_STEPS,
    EXPLORE_NAME,
    GENERAL_MAX_STEPS,
    GENERAL_NAME,
)
from .spec import SubAgentSpec


def general_subagent(
    llm: LLM,
    name: str = GENERAL_NAME,
    max_steps: int = GENERAL_MAX_STEPS,
    memories: list[Memory] | None = None,
) -> SubAgentSpec:
    """Return a general-purpose subagent spec.

    Equipped with ``DEFAULT_TOOLS``. Suited to open-ended, multi-step,
    context-heavy side work — the same profile as the coordinator minus
    the dispatch tool (depth 1) and a disposable per-dispatch context.

    Args:
        llm (LLM): Backbone LLM for the subagent.
        name (str, optional): Registry key / dispatch enum value.
            Defaults to `GENERAL_NAME`.
        max_steps (int, optional): Cap on steps per dispatch. Defaults
            to `GENERAL_MAX_STEPS`.
        memories (list[Memory] | None, optional): Memory backends for
            the subagent. Defaults to None.

    Returns:
        SubAgentSpec: A spec wrapping a `DEFAULT_TOOLS`-equipped agent.
    """
    return SubAgentSpec(
        name=name,
        description=(
            "General-purpose subagent for open-ended research, "
            "multi-step tasks, or work that benefits from an isolated "
            "context. Equipped with file reading and Python execution."
        ),
        agent=LLMAgent(llm=llm, tools=DEFAULT_TOOLS, memories=memories),
        max_steps=max_steps,
    )


def explore_subagent(
    llm: LLM,
    name: str = EXPLORE_NAME,
    max_steps: int = EXPLORE_MAX_STEPS,
) -> SubAgentSpec:
    """Return a read-only, lookup-tuned subagent spec.

    Equipped with only ``ReadFileTool`` — read-only by construction:
    the tool subset itself is the permission boundary, no separate
    permission system required. A cheap, fast option for lookups,
    well suited to a smaller model than the coordinator.

    Args:
        llm (LLM): Backbone LLM for the subagent.
        name (str, optional): Registry key / dispatch enum value.
            Defaults to `EXPLORE_NAME`.
        max_steps (int, optional): Cap on steps per dispatch. Lower
            than `general_subagent`'s default since lookups need
            fewer steps. Defaults to `EXPLORE_MAX_STEPS`.

    Returns:
        SubAgentSpec: A spec wrapping a `ReadFileTool`-only agent.
    """
    return SubAgentSpec(
        name=name,
        description=(
            "Read-only subagent for quickly finding and reading files. "
            "Use for lookups and fact-finding, not multi-step work."
        ),
        agent=LLMAgent(llm=llm, tools=[ReadFileTool()]),
        max_steps=max_steps,
    )
