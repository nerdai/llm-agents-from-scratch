"""A2AAgentSpec — specification for a registered A2A peer agent."""

from __future__ import annotations

import httpx
from a2a.client import A2ACardResolver
from a2a.types import AgentCard
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from llm_agents_from_scratch.errors import A2AAgentCardMissingInterfaceError

from .constants import (
    CATALOG_A2A_SKILL_TEMPLATE,
    CATALOG_A2A_SKILLS_TEMPLATE,
    CATALOG_SPEC_TEMPLATE,
)


class A2AAgentSpec(BaseModel):
    """Specification for a registered A2A peer agent.

    Each ``A2AAgentSpec`` entry registers a remote A2A-compliant peer under
    ``name``, derived directly from ``agent_card.name`` — there is no
    separate local alias, since the card is remote/peer-controlled data
    like the rest of the spec's inputs. ``name`` serves as the registry key
    on the coordinator and as the enum value ``UseA2AAgentTool`` presents
    to the LLM for dispatch. Discovery (fetching the peer's ``AgentCard``)
    is eager, at spec construction — the spec holds a fully resolved card,
    not a lazy reference to one.

    The spec is pure data: it never constructs or holds a live SDK
    ``Client``. Connecting to the peer is ``UseA2AAgentTool``'s job, done
    fresh on each dispatch from this spec's ``url``/``auth_headers``/
    ``agent_card``.

    ``url`` is likewise derived from the card rather than passed
    independently: it should match
    ``agent_card.supported_interfaces[0].url``, not the URL ``from_url``
    fetched the card from — the two can legitimately differ (that's the
    reason ``supported_interfaces`` exists as a separate list rather than
    a single top-level field). A card declaring more than one interface is
    a real possibility the protocol allows for, but this spec doesn't
    attempt to disambiguate between them — it always takes the first. Our
    own server (``LLMAgentA2AExecutor``, see #787) only ever publishes
    one, so this is a deliberate simplification, not an oversight.

    Attributes:
        name: Registry key for this A2A agent, taken from
            ``agent_card.name``. Appears as an enum value in
            ``UseA2AAgentTool``'s dispatch schema.
        url: Base URL of the remote A2A peer. Should match
            ``agent_card.supported_interfaces[0].url``.
        auth_headers: Optional auth headers (e.g. Authorization) sent on
            requests to this peer, both for card resolution and for
            dispatch.
        agent_card: The peer's resolved ``AgentCard``, fetched eagerly at
            construction time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        description=(
            "Registry key for this A2A agent, taken from agent_card.name. "
            "Appears as an enum value in UseA2AAgentTool's dispatch "
            "schema."
        ),
    )
    url: str = Field(
        description=(
            "Base URL of the remote A2A peer. Should match "
            "agent_card.supported_interfaces[0].url."
        ),
    )
    auth_headers: dict[str, SecretStr] | None = Field(
        default=None,
        description=(
            "Optional auth headers (e.g. Authorization) sent on requests "
            "to this peer, both for card resolution and for dispatch. "
            "Values are SecretStr: masked in repr(), model_dump(), and "
            "model_dump_json() alike. Call sites that hand these to "
            "httpx must unwrap with get_secret_value()."
        ),
    )
    agent_card: AgentCard = Field(
        description="The peer's resolved AgentCard.",
    )

    @classmethod
    def from_agent_card(
        cls,
        agent_card: AgentCard,
        auth_headers: dict[str, str] | None = None,
    ) -> A2AAgentSpec:
        """Builds a spec from an ``AgentCard`` already in hand.

        Sync — covers cached cards, self-built cards, and test fixtures,
        with no network access performed here.

        Args:
            agent_card: The peer's already-resolved ``AgentCard``.
            auth_headers: Optional auth headers sent on requests to this
                peer.

        Returns:
            A2AAgentSpec: The constructed spec.

        Raises:
            A2AAgentCardMissingInterfaceError: If ``agent_card`` declares
                no ``supported_interfaces`` to derive a dispatch URL from.
        """
        if not agent_card.supported_interfaces:
            raise A2AAgentCardMissingInterfaceError(
                f"AgentCard '{agent_card.name}' declares no "
                "supported_interfaces; cannot determine a dispatch URL.",
            )
        return cls(
            name=agent_card.name,
            url=agent_card.supported_interfaces[0].url,
            agent_card=agent_card,
            auth_headers=auth_headers,
        )

    @classmethod
    async def from_url(
        cls,
        url: str,
        auth_headers: dict[str, str] | None = None,
        agent_card_path: str | None = None,
    ) -> A2AAgentSpec:
        """Fetches the peer's ``AgentCard`` from ``url``, then builds a spec.

        Async — resolves the card over the wire via ``A2ACardResolver``
        before delegating to ``from_agent_card``. An unreachable peer
        raises the underlying ``httpx`` error to the caller. ``url`` here
        is only the card-resolution endpoint — the constructed spec's own
        ``url`` comes from the fetched card's ``supported_interfaces``
        instead, which can legitimately differ.

        Args:
            url: Base URL to resolve the peer's well-known ``AgentCard``
                from.
            auth_headers: Optional auth headers sent on requests to this
                peer.
            agent_card_path: Optional override for the well-known agent
                card path. ``None`` uses the SDK's own default.

        Returns:
            A2AAgentSpec: The constructed spec.

        Raises:
            A2AAgentCardMissingInterfaceError: If the fetched card
                declares no ``supported_interfaces``.
        """
        async with httpx.AsyncClient(headers=auth_headers) as httpx_client:
            resolver_kwargs: dict[str, str] = {}
            if agent_card_path is not None:
                resolver_kwargs["agent_card_path"] = agent_card_path
            resolver = A2ACardResolver(
                httpx_client=httpx_client,
                base_url=url,
                **resolver_kwargs,
            )
            agent_card = await resolver.get_agent_card()

        return cls.from_agent_card(
            agent_card=agent_card,
            auth_headers=auth_headers,
        )

    def catalog(self) -> str:
        """Returns XML structured string for cataloging this A2A agent.

        Nests the peer's declared ``AgentSkill``s (its ``agent_card.skills``)
        as an ``<a2a_skills>`` block, giving the coordinator finer-grained
        routing signal than the top-level description alone. Omitted
        entirely when the peer declares no skills.
        """
        skills = "\n".join(
            CATALOG_A2A_SKILL_TEMPLATE.format(name=skill.name)
            for skill in self.agent_card.skills
        )
        skills_block = (
            CATALOG_A2A_SKILLS_TEMPLATE.format(skills=skills) if skills else ""
        )
        return CATALOG_SPEC_TEMPLATE.format(
            name=self.name,
            description=self.agent_card.description,
            skills=skills_block,
        )
