"""Unit tests for A2AAgentSpec."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from a2a.types import AgentCard, AgentInterface, AgentSkill

from llm_agents_from_scratch.a2a import A2AAgentSpec
from llm_agents_from_scratch.errors import A2AAgentCardMissingInterfaceError

DISPATCH_URL = "http://127.0.0.1:9999"


def _agent_card(
    name: str = "peer",
    description: str = "does things",
    skills: list[AgentSkill] | None = None,
    supported_interfaces: list[AgentInterface] | None = None,
) -> AgentCard:
    if supported_interfaces is None:
        supported_interfaces = [AgentInterface(url=DISPATCH_URL)]
    return AgentCard(
        name=name,
        description=description,
        skills=skills or [],
        supported_interfaces=supported_interfaces,
    )


def test_a2aagentspec_from_agent_card() -> None:
    """Tests A2AAgentSpec.from_agent_card builds a spec with no I/O."""
    card = _agent_card(name="researcher")
    spec = A2AAgentSpec.from_agent_card(agent_card=card)

    assert spec.name == "researcher"
    assert spec.url == DISPATCH_URL
    assert spec.agent_card == card
    assert spec.headers is None


def test_a2aagentspec_from_agent_card_with_headers() -> None:
    """Tests A2AAgentSpec.from_agent_card stores explicit headers."""
    card = _agent_card()
    spec = A2AAgentSpec.from_agent_card(
        agent_card=card,
        headers={"Authorization": "Bearer token"},
    )

    assert spec.headers == {"Authorization": "Bearer token"}


def test_a2aagentspec_from_agent_card_uses_first_interface() -> None:
    """Tests url is taken from the first of multiple supported_interfaces."""
    card = _agent_card(
        supported_interfaces=[
            AgentInterface(url="http://first:9999"),
            AgentInterface(url="http://second:9999"),
        ],
    )
    spec = A2AAgentSpec.from_agent_card(agent_card=card)

    assert spec.url == "http://first:9999"


def test_a2aagentspec_from_agent_card_raises_on_no_interfaces() -> None:
    """Tests a card with no supported_interfaces raises a clear error."""
    card = _agent_card(supported_interfaces=[])

    with pytest.raises(
        A2AAgentCardMissingInterfaceError,
        match="declares no supported_interfaces",
    ):
        A2AAgentSpec.from_agent_card(agent_card=card)


def test_a2aagentspec_has_no_client_field() -> None:
    """Tests A2AAgentSpec never holds a live SDK client (Decision #783)."""
    assert set(A2AAgentSpec.model_fields.keys()) == {
        "name",
        "url",
        "headers",
        "agent_card",
    }


@pytest.mark.asyncio
async def test_a2aagentspec_from_url() -> None:
    """Tests A2AAgentSpec.from_url resolves the card then delegates."""
    card = _agent_card(name="researcher")

    with patch(
        "llm_agents_from_scratch.a2a.spec.A2ACardResolver.get_agent_card",
        new_callable=AsyncMock,
        return_value=card,
    ) as mock_get_agent_card:
        spec = await A2AAgentSpec.from_url(url="http://resolve-here:8888")

    mock_get_agent_card.assert_awaited_once()
    assert spec.name == "researcher"
    assert spec.agent_card == card
    # spec.url comes from the card's own interface, not the resolution url.
    assert spec.url == DISPATCH_URL


@pytest.mark.asyncio
async def test_a2aagentspec_from_url_passes_headers_and_card_path() -> None:
    """Tests A2AAgentSpec.from_url forwards headers and agent_card_path."""
    card = _agent_card()
    captured: dict[str, object] = {}

    def _capturing_init(self, *args: object, **kwargs: object) -> None:
        captured["kwargs"] = kwargs

    with (
        patch(
            "llm_agents_from_scratch.a2a.spec.A2ACardResolver.__init__",
            new=_capturing_init,
        ),
        patch(
            "llm_agents_from_scratch.a2a.spec.A2ACardResolver.get_agent_card",
            new_callable=AsyncMock,
            return_value=card,
        ),
    ):
        await A2AAgentSpec.from_url(
            url="http://127.0.0.1:9999",
            headers={"Authorization": "Bearer token"},
            agent_card_path="/custom/card.json",
        )

    assert captured["kwargs"]["base_url"] == "http://127.0.0.1:9999"
    assert captured["kwargs"]["agent_card_path"] == "/custom/card.json"


@pytest.mark.asyncio
async def test_a2aagentspec_from_url_propagates_connection_error() -> None:
    """Tests an unreachable peer raises to the caller (Decision #783/#8)."""
    with (
        patch(
            "llm_agents_from_scratch.a2a.spec.A2ACardResolver.get_agent_card",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("unreachable"),
        ),
        pytest.raises(httpx.ConnectError, match="unreachable"),
    ):
        await A2AAgentSpec.from_url(url="http://127.0.0.1:9999")


def test_a2aagentspec_catalog() -> None:
    """Tests A2AAgentSpec.catalog renders agent_card name and description."""
    spec = A2AAgentSpec.from_agent_card(
        agent_card=_agent_card(
            name="researcher",
            description="Searches the web.",
        ),
    )

    catalog = spec.catalog()

    assert "<name>researcher</name>" in catalog
    assert "<description>Searches the web.</description>" in catalog


def test_a2aagentspec_catalog_omits_skills_block_when_empty() -> None:
    """Tests catalog has no <a2a_skills> block when the card has none."""
    spec = A2AAgentSpec.from_agent_card(agent_card=_agent_card())

    assert "<a2a_skills>" not in spec.catalog()


def test_a2aagentspec_catalog_lists_skills() -> None:
    """Tests catalog nests each declared AgentSkill's name."""
    spec = A2AAgentSpec.from_agent_card(
        agent_card=_agent_card(
            skills=[
                AgentSkill(id="1", name="web_search", description="..."),
                AgentSkill(id="2", name="pdf_summarize", description="..."),
            ],
        ),
    )

    catalog = spec.catalog()

    assert "<a2a_skills>" in catalog
    assert "<a2a_skill>web_search</a2a_skill>" in catalog
    assert "<a2a_skill>pdf_summarize</a2a_skill>" in catalog
