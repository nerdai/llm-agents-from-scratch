"""Main application."""

import asyncio
import os
import re

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


def _extract_starting_integer(instruction: str) -> int | None:
    """Pulls the first integer named in a task instruction, if any.

    A deliberately simple, deterministic check rather than asking the
    CrewAI agent itself whether the instruction is ambiguous -- keeps
    the input_required path predictable for the notebook demo instead
    of depending on a free-text judgment call from the model.

    Args:
        instruction: The raw task instruction text.

    Returns:
        The first integer found, or None if the instruction names none.
    """
    match = re.search(r"\d+", instruction)
    return int(match.group()) if match else None


def _extract_step_budget(instruction: str) -> int | None:
    r"""Pulls a "run N steps" style step budget out of an instruction.

    Matched separately from ``_extract_starting_integer()`` (a bare
    ``\d+``) so a phrase like "starting at 8, run 3 steps then stop"
    doesn't confuse the two integers. Absent, ``execute()`` falls back
    to the original full-sequence-to-1 behavior, so every prior
    example's instruction (which never names a step count) is
    unaffected.

    Args:
        instruction: The raw task instruction text.

    Returns:
        The requested step count, or None if the instruction names
        none.
    """
    match = re.search(r"(\d+)\s+steps?", instruction)
    return int(match.group(1)) if match else None


@tool("hailstone_step")
def hailstone_step_fn(x: int) -> str:
    """Performs a single step of the Hailstone sequence.

    If x is even, returns x / 2. If x is odd, returns 3x + 1.
    """
    if x % 2 == 0:
        return str(x // 2)
    return str(3 * x + 1)


def build_crew(starting_value: int, step_budget: int | None = None) -> Crew:
    """Builds a single-agent Crew that computes a hailstone sequence.

    Args:
        starting_value: The positive integer to start the sequence
            from.
        step_budget: If given, the crew stops after exactly this many
            steps rather than continuing to 1 -- used by the relay
            demo in Chapter 10's Example 5, where this peer only
            carries the sequence partway before handing off to
            another agent. Defaults to None (run to completion).

    Returns:
        Crew: A crew ready to be kicked off.
    """
    llm = LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    if step_budget is None:
        goal = (
            "Compute the full Collatz/Hailstone sequence for the positive "
            "integer named in the task, by repeatedly calling the "
            "hailstone_step tool on each result until it reaches 1."
        )
        backstory = (
            "An expert on the Collatz conjecture who never computes a "
            "step by hand — always calls the hailstone_step tool once "
            "per step, feeding each result back in as the next call's "
            "input, until the sequence reaches 1."
        )
        description = (
            f"Compute the hailstone sequence starting at {starting_value}."
        )
        expected_output = (
            "The full hailstone sequence as a comma-separated list of "
            "integers, starting with the input value and ending at 1, "
            "and nothing else."
        )
    else:
        goal = (
            "Compute a partial Collatz/Hailstone sequence for the positive "
            "integer named in the task, by calling the hailstone_step "
            "tool exactly the number of times given, then stopping even "
            "if the sequence hasn't reached 1 yet."
        )
        backstory = (
            "An expert on the Collatz conjecture who never computes a "
            "step by hand — always calls the hailstone_step tool once "
            "per step, feeding each result back in as the next call's "
            "input, and stops as soon as the requested number of calls "
            "has been made."
        )
        description = (
            f"Starting at {starting_value}, call hailstone_step exactly "
            f"{step_budget} times, feeding each result into the next "
            f"call as input. Stop after exactly {step_budget} calls, "
            "even if the sequence hasn't reached 1 yet."
        )
        expected_output = (
            f"The sequence of {step_budget + 1} integers from "
            f"{starting_value} through the value after {step_budget} "
            "steps, as a comma-separated list, and nothing else."
        )
    agent = Agent(
        role="Hailstone Sequence Calculator",
        goal=goal,
        backstory=backstory,
        tools=[hailstone_step_fn],
        llm=llm,
    )
    task = Task(
        description=description,
        expected_output=expected_output,
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
        """Computes the full hailstone sequence for the caller's instruction.

        Asks for clarification (``TASK_STATE_INPUT_REQUIRED``) instead
        of guessing when the instruction names no starting integer --
        on the follow-up call, the resumed message's text (not the
        original instruction) is checked, since a peer only sees the
        newest message per call.

        Args:
            context: The request context containing the task instruction.
            event_queue: The queue to publish task status/artifact events
                to.
        """
        if context.current_task:
            task = context.current_task
        else:
            if context.message is None:
                raise ValueError(
                    "RequestContext is missing the user's Message.",
                )
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(
            event_queue,
            task.id,
            task.context_id,
        )
        await updater.submit()
        await updater.start_work()

        instruction = context.get_user_input()
        starting_value = _extract_starting_integer(instruction)
        if starting_value is None:
            await updater.requires_input(
                message=updater.new_agent_message(
                    [
                        new_text_part(
                            "Which positive integer should I start the "
                            "hailstone sequence from?",
                        ),
                    ],
                ),
            )
            return

        step_budget = _extract_step_budget(instruction)
        crew = build_crew(starting_value, step_budget)
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
                    "3x + 1 (if odd) until reaching 1. Asks for "
                    "clarification if the task doesn't name a starting "
                    "integer."
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
