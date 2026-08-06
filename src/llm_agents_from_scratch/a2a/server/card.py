"""build_agent_card — constructs an AgentCard for serving an LLMAgent."""

from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol


def build_agent_card(
    name: str,
    description: str,
    url: str,
    version: str = "0.1.0",
    skills: list[AgentSkill] | None = None,
) -> AgentCard:
    """Builds an ``AgentCard`` for serving an ``LLMAgent`` over A2A.

    A plain function returning the SDK's own type rather than a class
    of ours, so readers keep the protocol's vocabulary.

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

    Returns:
        AgentCard: The constructed card, with a single
            ``supported_interfaces`` entry advertising the JSON-RPC
            transport at ``url``, and streaming disabled — dispatch
            through this framework is always ``ClientConfig
            (streaming=False)`` on the client side (see
            ``UseA2AAgentTool``), so advertising streaming support here
            would be misleading.
    """
    return AgentCard(
        name=name,
        description=description,
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_1_0,
            ),
        ],
        version=version,
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=skills or [],
    )
