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
    capabilities: AgentCapabilities | None = None,
    provider: AgentProvider | None = None,
    documentation_url: str | None = None,
    security_schemes: dict[str, SecurityScheme] | None = None,
    security_requirements: list[SecurityRequirement] | None = None,
    signatures: list[AgentCardSignature] | None = None,
    icon_url: str | None = None,
) -> AgentCard:
    """Builds an ``AgentCard`` for serving an ``LLMAgent`` over A2A.

    A plain function returning the SDK's own type rather than a class
    of ours, so readers keep the protocol's vocabulary. Mirrors every
    ``AgentCard`` field except ``supported_interfaces`` and the
    default I/O modes, which this framework genuinely constrains
    rather than merely defaults: ``supported_interfaces`` is always a
    single JSON-RPC/v1.0 entry at ``url`` (the only transport
    ``DefaultRequestHandler``/``LLMAgentA2AExecutor`` speak), and
    ``default_input_modes``/``default_output_modes`` are always
    ``["text/plain"]`` (``LLMAgentA2AExecutor.execute()`` only extracts
    text via ``context.get_user_input()`` and only emits text via
    ``new_text_part``).

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
        capabilities (AgentCapabilities | None): The agent's declared
            capabilities. Defaults to ``AgentCapabilities(streaming=
            False)`` — ``LLMAgentA2AExecutor`` doesn't currently
            publish incremental status/artifact updates mid-task, so
            that's an honest default, not a restriction; pass your own
            ``AgentCapabilities`` to declare otherwise.
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
        # constrained by this framework's implementation, not passed thru as is
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_1_0,
            ),
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        # unconstrained -- mirrors AgentCard directly
        name=name,
        description=description,
        version=version,
        capabilities=capabilities or AgentCapabilities(streaming=False),
        skills=skills or [],
        provider=provider,
        documentation_url=documentation_url,
        security_schemes=security_schemes,
        security_requirements=security_requirements,
        signatures=signatures,
        icon_url=icon_url,
    )
