"""build_agent_card — constructs an AgentCard for serving an LLMAgent."""

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentCardSignature,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    SecurityRequirement,
    SecurityScheme,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol


def build_agent_card(  # noqa: PLR0913, PLR0917
    name: str,
    description: str,
    url: str,
    version: str = "0.1.0",
    skills: list[AgentSkill] | None = None,
    provider: AgentProvider | None = None,
    documentation_url: str | None = None,
    security_schemes: dict[str, SecurityScheme] | None = None,
    security_requirements: list[SecurityRequirement] | None = None,
    signatures: list[AgentCardSignature] | None = None,
    icon_url: str | None = None,
) -> AgentCard:
    """Builds an ``AgentCard`` for serving an ``LLMAgent`` over A2A.

    A plain function returning the SDK's own type rather than a class
    of ours, so readers keep the protocol's vocabulary. Necessarily
    opinionated, not a neutral general-purpose ``AgentCard``
    constructor: its job is a card that's honest about what
    ``LLMAgentA2AExecutor`` specifically does, so every field
    describing executor *behavior* is fixed rather than exposed as a
    parameter — ``supported_interfaces`` is always a single
    JSON-RPC/v1.0 entry at ``url`` (the only transport
    ``DefaultRequestHandler``/``LLMAgentA2AExecutor`` speak),
    ``default_input_modes``/``default_output_modes`` are always
    ``["text/plain"]`` (``execute()`` only extracts text via
    ``context.get_user_input()`` and only emits text via
    ``new_text_part``), and ``capabilities`` is always
    ``AgentCapabilities(streaming=False)`` (``execute()`` publishes
    only the final terminal state, no incremental updates — see the
    streaming executor variant tracked as a follow-up, issue #814).
    Everything else — pure descriptive metadata that doesn't claim
    anything about what the executor *does* — mirrors ``AgentCard``
    directly.

    Args:
        name (str): The agent's name, shown to peers.
        description (str): The agent's description, shown to peers.
        url (str): The deployment URL this agent will be served at.
            Must be supplied by the caller — nothing in this framework
            can infer where an ``LLMAgentA2AExecutor`` will actually be
            deployed.
        version (str): The agent's version string. Defaults to
            ``"0.1.0"``.
        skills (list[AgentSkill] | None): The agent's declared skills.
            Defaults to an empty list.
        provider (AgentProvider | None): The agent's provider
            organisation. Defaults to unset.
        documentation_url (str | None): URL to the agent's
            documentation. Defaults to unset.
        security_schemes (dict[str, SecurityScheme] | None): Named
            security schemes the agent supports. Defaults to none.
        security_requirements (list[SecurityRequirement] | None):
            Security requirements callers must satisfy. Defaults to
            none.
        signatures (list[AgentCardSignature] | None): Cryptographic
            signatures over the card. Defaults to none.
        icon_url (str | None): URL to an icon representing the agent.
            Defaults to unset.

    Returns:
        AgentCard: The constructed card, with a single
            ``supported_interfaces`` entry advertising the JSON-RPC
            transport at ``url``.
    """
    return AgentCard(
        # fixed: describes LLMAgentA2AExecutor's actual behavior, not
        # passed through as a parameter
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_1_0,
            ),
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        # descriptive metadata only -- mirrors AgentCard directly
        name=name,
        description=description,
        version=version,
        skills=skills or [],
        provider=provider,
        documentation_url=documentation_url,
        security_schemes=security_schemes,
        security_requirements=security_requirements,
        signatures=signatures,
        icon_url=icon_url,
    )
