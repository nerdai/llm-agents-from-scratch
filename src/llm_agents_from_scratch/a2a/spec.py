"""A2AAgentSpec — specification for a registered A2A peer agent."""

from __future__ import annotations

import httpx
from a2a.client import A2ACardResolver
from a2a.types import AgentCard
from pydantic import BaseModel, ConfigDict, Field

from .constants import CATALOG_SPEC_TEMPLATE


class A2AAgentSpec(BaseModel):
    """Specification for a registered A2A peer agent.

    Each ``A2AAgentSpec`` entry registers a remote A2A-compliant peer under
    a human-readable local name. The name serves as the registry key on the
    coordinator and as the enum value ``UseA2AAgentTool`` presents to the
    LLM for dispatch. Discovery (fetching the peer's ``AgentCard``) is
    eager, at spec construction — the spec holds a fully resolved card, not
    a lazy reference to one.

    The spec is pure data: it never constructs or holds a live SDK
    ``Client``. Connecting to the peer is ``UseA2AAgentTool``'s job, done
    fresh on each dispatch from this spec's ``url``/``headers``/
    ``agent_card``.

    Attributes:
        name: Unique registry key for this A2A agent. Appears as an enum
            value in ``UseA2AAgentTool``'s dispatch schema — must be
            human-readable and stable.
        url: Base URL of the remote A2A peer.
        headers: Optional HTTP headers (e.g. auth) sent on requests to this
            peer, both for card resolution and for dispatch.
        agent_card: The peer's resolved ``AgentCard``, fetched eagerly at
            construction time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(
        description=(
            "Unique registry key for this A2A agent. Appears as an enum "
            "value in UseA2AAgentTool's dispatch schema."
        ),
    )
    url: str = Field(description="Base URL of the remote A2A peer.")
    headers: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional HTTP headers sent on requests to this peer, both "
            "for card resolution and for dispatch."
        ),
    )
    agent_card: AgentCard = Field(
        description="The peer's resolved AgentCard.",
    )

    @classmethod
    def from_agent_card(
        cls,
        name: str,
        url: str,
        agent_card: AgentCard,
        headers: dict[str, str] | None = None,
    ) -> A2AAgentSpec:
        """Builds a spec from an ``AgentCard`` already in hand.

        Sync — covers cached cards, self-built cards, and test fixtures,
        with no network access performed here.

        Args:
            name: Unique registry key for this A2A agent.
            url: Base URL of the remote A2A peer.
            agent_card: The peer's already-resolved ``AgentCard``.
            headers: Optional HTTP headers sent on requests to this peer.

        Returns:
            A2AAgentSpec: The constructed spec.
        """
        return cls(
            name=name,
            url=url,
            agent_card=agent_card,
            headers=headers,
        )

    @classmethod
    async def from_url(
        cls,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        agent_card_path: str | None = None,
    ) -> A2AAgentSpec:
        """Fetches the peer's ``AgentCard`` from ``url``, then builds a spec.

        Async — resolves the card over the wire via ``A2ACardResolver``
        before delegating to ``from_agent_card``. An unreachable peer
        raises the underlying ``httpx`` error to the caller.

        Args:
            name: Unique registry key for this A2A agent.
            url: Base URL of the remote A2A peer.
            headers: Optional HTTP headers sent on requests to this peer.
            agent_card_path: Optional override for the well-known agent
                card path. ``None`` uses the SDK's own default.

        Returns:
            A2AAgentSpec: The constructed spec.
        """
        async with httpx.AsyncClient(headers=headers) as httpx_client:
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
            name=name,
            url=url,
            agent_card=agent_card,
            headers=headers,
        )

    def catalog(self) -> str:
        """Returns XML structured string for cataloging this A2A agent."""
        return CATALOG_SPEC_TEMPLATE.format(
            name=self.name,
            description=self.agent_card.description,
        )
