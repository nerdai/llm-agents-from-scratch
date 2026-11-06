import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

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


def _write_notebook(
    path: Path,
    metadata: dict[str, Any],
    source: str = "x = 1\n",
) -> Path:
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "source": [source],
                "outputs": [],
                "execution_count": None,
            },
        ],
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def test_status_settings_absent(run_notebook: ModuleType) -> None:
    assert run_notebook._status_settings({"metadata": {}}) == (None, None)
    assert run_notebook._status_settings({}) == (None, None)


def test_status_settings(run_notebook: ModuleType) -> None:
    nb = {
        "metadata": {
            "notebook_status": {
                "timeout": 1800,
                "providers": ["ollama-cloud"],
            },
        },
    }

    assert run_notebook._status_settings(nb) == (1800.0, ("ollama-cloud",))


@pytest.mark.parametrize(
    "settings",
    [
        ["ollama-cloud"],
        {"timeout": 0},
        {"timeout": "1800"},
        {"timeout": True},
        # too large for a float: float() would raise OverflowError
        {"timeout": 10**400},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
        {"providers": "ollama-cloud"},
        {"providers": ["ollama"]},
        {"timout": 1800},
    ],
)
def test_status_settings_malformed(
    run_notebook: ModuleType,
    settings: Any,
) -> None:
    nb = {"metadata": {"notebook_status": settings}}

    with pytest.raises(ValueError, match="notebook_status"):
        run_notebook._status_settings(nb)


def test_run_one_opted_out(
    run_notebook: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_notebook(
        tmp_path / "nb.ipynb",
        {"notebook_status": {"providers": ["ollama-cloud"]}},
        "llm = make_llm()\n",
    )
    monkeypatch.setattr(
        run_notebook,
        "_execute",
        lambda *a, **k: pytest.fail("an opted-out notebook was run"),
    )

    entry = run_notebook.run_one(path, "openai", 600, keep_outputs=False)

    assert entry["status"] == "opted-out"
    assert entry["seconds"] == 0
    assert "ollama-cloud" in entry["note"]


def test_run_one_malformed_metadata_fails(
    run_notebook: ModuleType,
    tmp_path: Path,
) -> None:
    path = _write_notebook(
        tmp_path / "nb.ipynb",
        {"notebook_status": {"providers": ["ollama"]}},
    )

    entry = run_notebook.run_one(path, "openai", 600, keep_outputs=False)

    assert entry["status"] == "fail"
    assert "notebook_status" in entry["note"]


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, 600),
        ({"notebook_status": {"timeout": 1800}}, 1800),
        # a listed provider still runs, with the notebook's timeout
        (
            {
                "notebook_status": {
                    "timeout": 90,
                    "providers": ["ollama-cloud"],
                },
            },
            90,
        ),
    ],
)
def test_run_one_timeout(  # noqa: PLR0913 - fixtures plus parameters
    run_notebook: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, Any],
    expected: float,
) -> None:
    path = _write_notebook(tmp_path / "nb.ipynb", metadata)
    seen: list[float] = []

    def fake_execute(
        path: Path,
        nb: dict[str, Any],
        provider: str,
        timeout: float,
        max_memory_gb: float | None = None,
    ) -> str | None:
        seen.append(timeout)
        return None

    monkeypatch.setattr(run_notebook, "_execute", fake_execute)

    # no LLM in the notebook, so it is only executed under Ollama Cloud
    entry = run_notebook.run_one(path, "ollama-cloud", 600, False)

    assert entry["status"] == "no-llm"
    assert seen == [expected]


def test_main_opted_out_is_not_a_regression(
    run_notebook: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ledger keys are paths relative to the repo
    monkeypatch.setattr(run_notebook, "REPO", tmp_path)
    path = _write_notebook(
        tmp_path / "nb.ipynb",
        {"notebook_status": {"providers": ["ollama-cloud"]}},
        "llm = make_llm()\n",
    )
    ledger = tmp_path / "ledger.yaml"
    ledger.write_text(
        yaml.safe_dump(
            {"nb.ipynb": {"openai": {"status": "pass", "seconds": 300}}},
        ),
        encoding="utf-8",
    )
    regressions = tmp_path / "regressions.json"

    with pytest.raises(SystemExit) as exit_info:
        run_notebook.main(
            str(path),
            provider="openai",
            ledger=str(ledger),
            regressions_out=str(regressions),
            budget_minutes=1,
        )

    assert exit_info.value.code == 0
    assert json.loads(regressions.read_text(encoding="utf-8")) == []
    data = yaml.safe_load(ledger.read_text(encoding="utf-8"))
    assert data["nb.ipynb"]["openai"]["status"] == "opted-out"


def test_main_budget_records_opted_out(
    run_notebook: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_notebook, "REPO", tmp_path)
    first = _write_notebook(tmp_path / "first.ipynb", {}, "llm = make_llm()\n")
    opted_out = _write_notebook(
        tmp_path / "opted_out.ipynb",
        {"notebook_status": {"providers": ["ollama-cloud"]}},
        "llm = make_llm()\n",
    )
    ledger = tmp_path / "ledger.yaml"
    ledger.write_text(
        yaml.safe_dump(
            {
                # stalest first, so first.ipynb is the run the budget
                # always lets through
                "first.ipynb": {
                    "openai": {"last_tested": "2026-01-01", "seconds": 10},
                },
                # its last run alone would overrun the budget
                "opted_out.ipynb": {
                    "openai": {
                        "status": "pass",
                        "last_tested": "2026-02-01",
                        "seconds": 300,
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    real_run_with_retries = run_notebook._run_with_retries

    def fake_run_with_retries(path: Path, *a: Any, **k: Any) -> Any:
        if path == first:
            return run_notebook._entry("pass", seconds=10)
        return real_run_with_retries(path, *a, **k)

    monkeypatch.setattr(
        run_notebook,
        "_run_with_retries",
        fake_run_with_retries,
    )

    with pytest.raises(SystemExit):
        run_notebook.main(
            str(first),
            str(opted_out),
            provider="openai",
            ledger=str(ledger),
            budget_minutes=1,
        )

    data = yaml.safe_load(ledger.read_text(encoding="utf-8"))
    assert data["opted_out.ipynb"]["openai"]["status"] == "opted-out"
