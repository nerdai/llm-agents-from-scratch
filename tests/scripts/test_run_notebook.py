import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

RUNNER = Path(__file__).parents[2] / "_scripts" / "run_notebook.py"


@pytest.fixture(scope="module")
def run_notebook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_notebook", RUNNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("text", "provider", "expected"),
    [
        # ch10 prints its A2A server's log, which reports Ollama Cloud
        # after the notebook's own make_llm() line
        (
            "✓ Using Anthropic (claude-opus-5)\n"
            "✓ Using Ollama Cloud (deepseek-v4.1-flash)\n",
            "anthropic",
            ("Anthropic", "claude-opus-5"),
        ),
        # ch03 reports ollama_settings() before the swapped make_llm()
        (
            "✓ Using Ollama Cloud (deepseek-v4.1-flash)\n"
            "✓ Using OpenAI (gpt-5)\n",
            "openai",
            ("OpenAI", "gpt-5"),
        ),
        # no line names the provider: the last one, so it's caught
        (
            "✓ Using Ollama (qwen3:14b)\n"
            "✓ Using Ollama Cloud (deepseek-v4.1-flash)\n",
            "anthropic",
            ("Ollama Cloud", "deepseek-v4.1-flash"),
        ),
    ],
)
def test_reported_llm(
    run_notebook: ModuleType,
    text: str,
    provider: str,
    expected: tuple[str, str],
) -> None:
    match = run_notebook._reported_llm(text, provider)

    assert match is not None
    assert match.groups() == expected


def test_reported_llm_none(run_notebook: ModuleType) -> None:
    assert run_notebook._reported_llm("no llm here\n", "anthropic") is None
