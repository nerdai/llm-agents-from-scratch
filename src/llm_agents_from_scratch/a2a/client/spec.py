"""A2AAgentSpec — specification for a registered A2A peer agent."""

from __future__ import annotations

import httpx
from a2a.client import A2ACardResolver
from a2a.types import AgentCard
from pydantic import BaseModel, ConfigDict, Field

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
    fresh on each dispatch from this spec's ``url``/``headers``/
    ``agent_card``.

    ``url`` is likewise derived from the card rather than passed
    independently: it should match
    ``agent_card.supported_interfaces[0].url``, not the URL ``from_url``
    fetched the card from — the two can legitimately differ (that's the
    reason ``supported_interfaces`` exists as a separate list rather than
    a single top-level field). A card declaring more than one interface is
    a real possibility the protocol allows for, but this spec doesn't
    attempt to disambiguate between them — it always takes the first. Our
    own server (``LLMAgentA2AExecutor``) only ever publishes one, so
    this is a deliberate simplification, not an oversight.

    Attributes:
        name: Registry key for this A2A agent, taken from
            ``agent_card.name``. Appears as an enum value in
            ``UseA2AAgentTool``'s dispatch schema.
        url: Base URL of the remote A2A peer. Should match
            ``agent_card.supported_interfaces[0].url``.
        headers: Optional HTTP headers (e.g. auth) sent on requests to
            this peer, both for card resolution and for dispatch, mirroring
            ``MCPToolProvider.streamable_http_headers``. May carry
            credentials (e.g. ``Authorization``); stored as a plain
            ``str``, not masked. Production use should wrap values in
            pydantic's ``SecretStr`` so a stray repr/log/dump can't leak
            them — left as a callout rather than built in, to keep this
            an educational framework rather than a production-hardened
            one.
        agent_card: The peer's resolved ``AgentCard``, fetched eagerly at
            construction time.
        timeout: Seconds ``UseA2AAgentTool`` allows a dispatch to this
            peer before timing out. Defaults to 60.0 rather than
            ``httpx``'s own default (5.0s, applied to connect/read/
            write/pool combined) — too low for a peer that makes one or
            more LLM calls per task. Explicitly setting this to
            ``None`` disables the timeout entirely (unbounded), since
            it is passed straight through to ``httpx.AsyncClient``.
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
    headers: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional HTTP headers sent on requests to this peer, both "
            "for card resolution and for dispatch. May carry credentials "
            "(e.g. Authorization); not masked — production use should "
            "wrap values in pydantic's SecretStr."
        ),
    )
    agent_card: AgentCard = Field(
        description="The peer's resolved AgentCard.",
    )
    timeout: float | None = Field(
        default=60.0,
        description=(
            "Seconds UseA2AAgentTool allows a dispatch to this peer "
            "before timing out. Explicitly setting this to None "
            "disables the timeout entirely (unbounded)."
        ),
    )

    @classmethod
    def from_agent_card(
        cls,
        agent_card: AgentCard,
        headers: dict[str, str] | None = None,
        timeout: float | None = 60.0,
    ) -> A2AAgentSpec:
        """Builds a spec from an ``AgentCard`` already in hand.

        Sync — covers cached cards, self-built cards, and test fixtures,
        with no network access performed here.

        Args:
            agent_card: The peer's already-resolved ``AgentCard``.
            headers: Optional HTTP headers sent on requests to this peer.
            timeout: Seconds ``UseA2AAgentTool`` allows a dispatch to
                this peer before timing out.

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
            headers=headers,
            timeout=timeout,
        )

    @classmethod
    async def from_url(
        cls,
        url: str,
        headers: dict[str, str] | None = None,
        agent_card_path: str | None = None,
        timeout: float | None = 60.0,
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
            headers: Optional HTTP headers sent on requests to this peer.
            agent_card_path: Optional override for the well-known agent
                card path. ``None`` uses the SDK's own default.
            timeout: Timeout in seconds, applied both to this card
                resolution and, via the returned spec, to every future
                ``UseA2AAgentTool`` dispatch to this peer.

        Returns:
            A2AAgentSpec: The constructed spec.

        Raises:
            A2AAgentCardMissingInterfaceError: If the fetched card
                declares no ``supported_interfaces``.
        """
        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
        ) as httpx_client:
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
            headers=headers,
            timeout=timeout,
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
