"""Unit tests for build_agent_card."""

import inspect

from a2a.types import (
    AgentCardSignature,
    AgentProvider,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol

from llm_agents_from_scratch.a2a.server.card import build_agent_card


def test_build_agent_card_defaults() -> None:
    """Tests defaults: version, empty skills, streaming disabled."""
    card = build_agent_card(
        name="my-agent",
        description="Does things.",
        url="http://localhost:9999",
    )

    assert card.name == "my-agent"
    assert card.description == "Does things."
    assert card.version == "0.1.0"
    assert list(card.skills) == []
    assert card.capabilities.streaming is False
    assert list(card.default_input_modes) == ["text/plain"]
    assert list(card.default_output_modes) == ["text/plain"]


def test_build_agent_card_supported_interface() -> None:
    """Tests the single supported_interfaces entry matches url/JSONRPC/1.0."""
    card = build_agent_card(
        name="my-agent",
        description="Does things.",
        url="http://localhost:9999",
    )

    assert len(card.supported_interfaces) == 1
    interface = card.supported_interfaces[0]
    assert interface.url == "http://localhost:9999"
    assert interface.protocol_binding == TransportProtocol.JSONRPC
    assert interface.protocol_version == PROTOCOL_VERSION_1_0


def test_build_agent_card_version_and_skills() -> None:
    """Tests explicit version and skills pass through."""
    skill = AgentSkill(
        id="hailstone",
        name="hailstone",
        description="Computes hailstone steps.",
        tags=["math"],
    )

    card = build_agent_card(
        name="my-agent",
        description="Does things.",
        url="http://localhost:9999",
        version="1.2.3",
        skills=[skill],
    )

    assert card.version == "1.2.3"
    assert list(card.skills) == [skill]


def test_build_agent_card_capabilities_not_a_parameter() -> None:
    """Tests capabilities can't be overridden -- it describes real behavior.

    Unlike pure metadata fields (provider, icon_url, ...),
    capabilities.streaming describes what LLMAgentA2AExecutor actually
    does. Letting a caller override it to streaming=True would produce
    a card that lies about the executor it's paired with.
    """
    assert "capabilities" not in inspect.signature(build_agent_card).parameters


def test_build_agent_card_metadata_fields_pass_through() -> None:
    """Tests provider/documentation_url/security/signatures/icon_url."""
    provider = AgentProvider(url="https://example.com", organization="Acme")
    scheme = SecurityScheme(
        http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="bearer"),
    )
    requirement = SecurityRequirement(schemes={})
    signature = AgentCardSignature(protected="hdr", signature="sig")

    card = build_agent_card(
        name="my-agent",
        description="Does things.",
        url="http://localhost:9999",
        provider=provider,
        documentation_url="https://example.com/docs",
        security_schemes={"bearer": scheme},
        security_requirements=[requirement],
        signatures=[signature],
        icon_url="https://example.com/icon.png",
    )

    assert card.provider == provider
    assert card.documentation_url == "https://example.com/docs"
    assert dict(card.security_schemes) == {"bearer": scheme}
    assert list(card.security_requirements) == [requirement]
    assert list(card.signatures) == [signature]
    assert card.icon_url == "https://example.com/icon.png"
