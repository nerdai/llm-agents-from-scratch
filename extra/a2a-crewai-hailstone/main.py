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
    """Builds a single-agent Crew that performs one hailstone step.

    Args:
        instruction: The raw task instruction from the A2A caller,
            expected to name the integer to step from.

    Returns:
        Crew: A crew ready to be kicked off with this instruction.
    """
    llm = LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    agent = Agent(
        role="Hailstone Sequence Calculator",
        goal=(
            "Compute a single step of the Collatz/Hailstone sequence for "
            "the integer named in the task."
        ),
        backstory=(
            "An expert on the Collatz conjecture who applies the "
            "hailstone step rule precisely, one step at a time, always "
            "using the hailstone_step tool rather than computing by hand."
        ),
        tools=[hailstone_step_fn],
        llm=llm,
    )
    task = Task(
        description=instruction,
        expected_output=(
            "The integer result of the hailstone step, and nothing else."
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
            "Computes one step of the Hailstone (Collatz) sequence via a "
            "CrewAI agent."
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
                id="hailstone_step",
                name="hailstone_step",
                description=(
                    "Compute one hailstone step for an integer x: x / 2 "
                    "if even, else 3x + 1."
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
    uvicorn.run(app, host=A2A_HOST, port=A2A_PORT)  # pragma: no cover
