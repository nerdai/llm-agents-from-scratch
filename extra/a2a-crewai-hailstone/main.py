"""Main application."""

import asyncio
import os

import uvicorn
from a2a.helpers import new_task_from_user_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol
from a2a.utils.errors import UnsupportedOperationError
from crewai import LLM, Agent, Crew, Task
from crewai.tools import tool
from fastapi import FastAPI

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "ollama/qwen3:14b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
A2A_HOST = os.environ.get("A2A_HOST", "0.0.0.0")
A2A_PORT = int(os.environ.get("A2A_PORT", "9200"))
A2A_URL = os.environ.get("A2A_URL", f"http://localhost:{A2A_PORT}")


@tool("hailstone_step")
def hailstone_step_fn(x: int) -> str:
    """Performs a single step of the Hailstone sequence.

    If x is even, returns x / 2. If x is odd, returns 3x + 1.
    """
    if x % 2 == 0:
        return str(x // 2)
    return str(3 * x + 1)


def build_crew(instruction: str) -> Crew:
    """Builds a single-agent Crew that computes a full hailstone sequence.

    Args:
        instruction: The raw task instruction from the A2A caller,
            expected to name the positive integer to start from.

    Returns:
        Crew: A crew ready to be kicked off with this instruction.
    """
    llm = LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    agent = Agent(
        role="Hailstone Sequence Calculator",
        goal=(
            "Compute the full Collatz/Hailstone sequence for the positive "
            "integer named in the task, by repeatedly calling the "
            "hailstone_step tool on each result until it reaches 1."
        ),
        backstory=(
            "An expert on the Collatz conjecture who never computes a "
            "step by hand — always calls the hailstone_step tool once "
            "per step, feeding each result back in as the next call's "
            "input, until the sequence reaches 1."
        ),
        tools=[hailstone_step_fn],
        llm=llm,
    )
    task = Task(
        description=instruction,
        expected_output=(
            "The full hailstone sequence as a comma-separated list of "
            "integers, starting with the input value and ending at 1, "
            "and nothing else."
        ),
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task])


class CrewAIHailstoneExecutor(AgentExecutor):
    """Bridges incoming A2A tasks to the CrewAI Hailstone crew."""

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Runs one hailstone step for the caller's instruction.

        Args:
            context: The request context containing the task instruction.
            event_queue: The queue to publish task status/artifact events
                to.
        """
        if context.task_id is None or context.context_id is None:
            raise ValueError(
                "RequestContext is missing a task_id or context_id.",
            )
        if context.current_task is None:
            if context.message is None:
                raise ValueError(
                    "RequestContext is missing the user's Message.",
                )
            await event_queue.enqueue_event(
                new_task_from_user_message(context.message),
            )
        updater = TaskUpdater(
            event_queue,
            context.task_id,
            context.context_id,
        )
        await updater.submit()
        await updater.start_work()

        instruction = context.get_user_input()
        crew = build_crew(instruction)
        result = await asyncio.to_thread(crew.kickoff)

        await updater.add_artifact(
            parts=[new_text_part(result.raw)],
            name="hailstone_result",
        )
        await updater.complete()

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancellation is not supported by this simple agent."""
        raise UnsupportedOperationError(
            "CrewAI Hailstone agent does not support cancellation.",
        )


def build_app() -> FastAPI:
    """Builds the FastAPI app serving the CrewAI Hailstone agent over A2A.

    Returns:
        FastAPI: The app, with A2A agent-card, JSON-RPC, and REST routes
            mounted.
    """
    agent_card = AgentCard(
        name="crewai-hailstone",
        description=(
            "Computes the full Hailstone (Collatz) sequence for a "
            "positive integer via a CrewAI agent."
        ),
        supported_interfaces=[
            AgentInterface(
                url=A2A_URL,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_1_0,
            ),
        ],
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[
            AgentSkill(
                id="hailstone_sequence",
                name="hailstone_sequence",
                description=(
                    "Compute the full hailstone sequence for a positive "
                    "integer x, repeatedly applying x / 2 (if even) or "
                    "3x + 1 (if odd) until reaching 1."
                ),
                tags=["math"],
            ),
        ],
    )
    request_handler = DefaultRequestHandler(
        agent_executor=CrewAIHailstoneExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    app = FastAPI()
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card),
        jsonrpc_routes=create_jsonrpc_routes(request_handler, rpc_url="/"),
        rest_routes=create_rest_routes(request_handler),
    )
    return app


app = build_app()


if __name__ == "__main__":
    uvicorn.run(app, host=A2A_HOST, port=A2A_PORT)
