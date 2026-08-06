"""Unit tests for A2AAgentSpec."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from a2a.types import AgentCard

from llm_agents_from_scratch.a2a import A2AAgentSpec


def _agent_card(
    name: str = "peer",
    description: str = "does things",
) -> AgentCard:
    return AgentCard(name=name, description=description)


def test_a2aagentspec_from_agent_card() -> None:
    """Tests A2AAgentSpec.from_agent_card builds a spec with no I/O."""
    card = _agent_card()
    spec = A2AAgentSpec.from_agent_card(
        name="researcher",
        url="http://127.0.0.1:9999",
        agent_card=card,
    )

    assert spec.name == "researcher"
    assert spec.url == "http://127.0.0.1:9999"
    assert spec.agent_card == card
    assert spec.headers is None


def test_a2aagentspec_from_agent_card_with_headers() -> None:
    """Tests A2AAgentSpec.from_agent_card stores explicit headers."""
    card = _agent_card()
    spec = A2AAgentSpec.from_agent_card(
        name="researcher",
        url="http://127.0.0.1:9999",
        agent_card=card,
        headers={"Authorization": "Bearer token"},
    )

    assert spec.headers == {"Authorization": "Bearer token"}


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
    card = _agent_card()

    with patch(
        "llm_agents_from_scratch.a2a.spec.A2ACardResolver.get_agent_card",
        new_callable=AsyncMock,
        return_value=card,
    ) as mock_get_agent_card:
        spec = await A2AAgentSpec.from_url(
            name="researcher",
            url="http://127.0.0.1:9999",
        )

    mock_get_agent_card.assert_awaited_once()
    assert spec.name == "researcher"
    assert spec.url == "http://127.0.0.1:9999"
    assert spec.agent_card == card


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
            name="researcher",
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
        await A2AAgentSpec.from_url(
            name="researcher",
            url="http://127.0.0.1:9999",
        )


def test_a2aagentspec_catalog() -> None:
    """Tests A2AAgentSpec.catalog renders name and card description."""
    spec = A2AAgentSpec.from_agent_card(
        name="researcher",
        url="http://127.0.0.1:9999",
        agent_card=_agent_card(
            name="ignored-card-name",
            description="Searches the web.",
        ),
    )

    catalog = spec.catalog()

    assert "<name>researcher</name>" in catalog
    assert "<description>Searches the web.</description>" in catalog
