"""Main application."""

import os

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from fastapi import FastAPI

from llm_agents_from_scratch import LLMAgent
from llm_agents_from_scratch.a2a import LLMAgentA2AExecutor, build_agent_card
from llm_agents_from_scratch.llms import OllamaLLM
from llm_agents_from_scratch.logger import enable_console_logging
from llm_agents_from_scratch.tools import SimpleFunctionTool

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST") or None
A2A_HOST = os.environ.get("A2A_HOST", "0.0.0.0")
A2A_PORT = int(os.environ.get("A2A_PORT", "9300"))
A2A_URL = os.environ.get("A2A_URL", f"http://localhost:{A2A_PORT}")

enable_console_logging()


def next_number(x: int) -> int:
    """Performs a single step of the Hailstone sequence.

    If x is even, returns x / 2. If x is odd, returns 3x + 1.
    """
    if x % 2 == 0:
        return x // 2
    return 3 * x + 1


def build_app() -> FastAPI:
    """Builds the FastAPI app serving an LLMAgent over A2A.

    The exact same executor/card construction pattern demonstrated
    inline in Example 4a of examples/ch10.ipynb -- this app is what
    actually serves it.

    Returns:
        FastAPI: The app, with A2A agent-card, JSON-RPC, and REST
            routes mounted.
    """
    # Exact match, not a prefix/substring check -- OLLAMA_HOST is only
    # ever unset (local) or exactly "https://ollama.com" per this
    # app's README, and a substring/prefix check would incorrectly
    # match a host like "https://ollama.com.evil.com".
    json_prompt_mode = OLLAMA_HOST == "https://ollama.com"
    llm = OllamaLLM(
        host=OLLAMA_HOST,
        model=OLLAMA_MODEL,
        think=False,
        json_prompt_mode=json_prompt_mode,
    )
    agent = LLMAgent(llm=llm, tools=[SimpleFunctionTool(func=next_number)])
    card = build_agent_card(
        name="from-scratch-hailstone",
        description=(
            "Computes the full Hailstone (Collatz) sequence for a "
            "positive integer."
        ),
        url=A2A_URL,
    )
    request_handler = DefaultRequestHandler(
        agent_executor=LLMAgentA2AExecutor(agent=agent),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    app = FastAPI()
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(request_handler, rpc_url="/"),
        rest_routes=create_rest_routes(request_handler),
    )
    return app


app = build_app()


if __name__ == "__main__":
    uvicorn.run(app, host=A2A_HOST, port=A2A_PORT)
