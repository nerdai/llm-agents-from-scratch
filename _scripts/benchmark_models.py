r"""Benchmark Ollama Cloud models on what the chapter notebooks teach.

Passing is not enough: a model can run a notebook to completion while
faking a tool call, skipping the skill the chapter is about, or letting
a subagent run out of steps. This runs chapter notebooks in memory with
`OLLAMA_MODEL=<model>` (nothing is written to the repo or the ledger) and
records, per run, whether the notebook *ran* and whether it *behaved*:
ran, and met every expectation in `EXPECTATIONS` for that chapter.

Runs use an isolated HOME by default, so user-level skills (e.g. in
~/.claude/skills) don't leak into the agents' skill catalog; uv's cache
and managed Pythons still point at the real HOME.

Commands:

    models    list the cloud models available to OLLAMA_API_KEY
    run       benchmark models on chapters, appending JSON lines
    summary   rank the models in a results file

Usage:

    uv run python _scripts/benchmark_models.py models
    uv run python _scripts/benchmark_models.py run gemma4:31b glm-5.2 \
        --chapters ch03,ch04,ch05,ch06,ch07 --runs 1 --out bench.jsonl
    uv run python _scripts/benchmark_models.py summary bench.jsonl

The project skill `.claude/skills/benchmark-ollama-models` describes when
to run this and how to read the results.
"""

import collections
import contextlib
import json
import os
import re
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import fire
import run_notebook as rn

REPO = Path(__file__).resolve().parents[1]
CHAPTERS = REPO / "chapter-notebooks"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
#: One agent's log line: an optional "[subagent] " prefix, then the event.
LOG = re.compile(
    r"^(?P<who>\[[\w-]+\] )?.*?"
    r"(?:Executing Tool Call: (?P<tool>[\w.-]+)"
    r"|(?P<ok>✅ Successful Tool Call)"
    r"|(?P<err>❌ Tool Call Failure))",
)
ROLLOUT = re.compile(r"BENCH_TOOL_RESULTS (\d+)")
#: ch04 starts its agent in one cell and reads it later; count the
#: *successful* tool results in the agent's rollout instead of its logs.
ROLLOUT_PROBE = (
    "print('BENCH_TOOL_RESULTS', sum("
    "1 for line in handler.rollout.splitlines() if line.startswith('🔧')"
    " and 'error_type' not in line and \"doesn't exist\" not in line))"
)


def _tools(text: str) -> collections.Counter[str]:
    """Count *successful* tool calls per tool name.

    The agent logs "Executing Tool Call: <name>" before it runs a tool,
    then "Successful Tool Call" or "Tool Call Failure", so an attempt alone
    proves nothing. Results are matched to attempts per agent (subagents
    prefix their lines with "[name] ") in arrival order. A step's calls run
    concurrently, so this is exact as long as one step's calls are to the
    same tool, which holds for the chapter notebooks.
    """
    pending: dict[str, list[str]] = collections.defaultdict(list)
    ok: collections.Counter[str] = collections.Counter()
    for line in text.splitlines():
        m = LOG.search(line)
        if not m:
            continue
        who = m.group("who") or ""
        if m.group("tool"):
            pending[who].append(m.group("tool"))
        elif pending[who]:
            tool = pending[who].pop(0)
            if m.group("ok"):
                ok[tool] += 1
    return ok


def _rollout_tools(text: str) -> int:
    m = ROLLOUT.search(text)
    return int(m.group(1)) if m else 0


Check = Callable[[str], bool]
#: What each chapter's agents must do, as (description, check on the run's
#: outputs). Add a row to tighten a chapter; a chapter with no rows only
#: has to run.
EXPECTATIONS: dict[str, list[tuple[str, Check]]] = {
    "ch04": [
        (
            "agent made >=2 real tool calls",
            lambda t: _rollout_tools(t) >= 2,  # noqa: PLR2004
        ),
    ],
    "ch05": [
        (
            "called the MCP hailstone tool",
            lambda t: _tools(t)["mcp__hailstone__hailstone_step_fn"] >= 1,
        ),
    ],
    "ch06": [
        (
            "activated the skill",
            lambda t: _tools(t)["from_scratch__use_skill"] >= 1,
        ),
        ("called next_number", lambda t: _tools(t)["next_number"] >= 1),
    ],
    "ch07": [
        (
            "activated the skill",
            lambda t: _tools(t)["from_scratch__use_skill"] >= 1,
        ),
        ("called next_number", lambda t: _tools(t)["next_number"] >= 1),
    ],
    "ch08": [
        (
            "asked the human",
            lambda t: _tools(t)["from_scratch__human_input"] >= 1,
        ),
        (
            "called next_number >=3 times",
            lambda t: _tools(t)["next_number"] >= 3,  # noqa: PLR2004
        ),
    ],
    "ch09": [
        (
            "no subagent hit max steps",
            lambda t: "MaxStepsReachedError" not in t,
        ),
        (
            ">=5 subagent dispatches",
            lambda t: _tools(t)["from_scratch__use_subagent"] >= 5,  # noqa: PLR2004
        ),
        (
            "asked the human twice",
            lambda t: _tools(t)["from_scratch__human_input"] >= 2,  # noqa: PLR2004
        ),
    ],
}


def _outputs(nb: dict[str, Any]) -> str:
    text = ""
    for cell in nb["cells"]:
        for o in cell.get("outputs", []):
            text += "".join(o.get("text", ""))
            text += "".join(o.get("data", {}).get("text/plain", ""))
            if o.get("output_type") == "error":
                text += f"\nERROR {o.get('ename')}: {o.get('evalue')}"
    return ANSI.sub("", text)


@contextlib.contextmanager
def _isolated_home() -> Iterator[None]:
    """Point HOME at an empty temp dir for the duration, then clean up."""
    real = Path.home()
    saved = {k: os.environ.get(k) for k in _HOME_VARS}
    os.environ.setdefault("UV_CACHE_DIR", str(real / ".cache" / "uv"))
    os.environ.setdefault(
        "UV_PYTHON_INSTALL_DIR",
        str(real / ".local" / "share" / "uv" / "python"),
    )
    with tempfile.TemporaryDirectory(prefix="bench-home-") as home:
        os.environ["HOME"] = home
        try:
            yield
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


_HOME_VARS = ("HOME", "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR")


def _run_once(model: str, chapter: str, timeout: float) -> dict[str, Any]:
    # per-chapter folders (chapter-notebooks/chXX/chXX.ipynb) or flat
    nested = CHAPTERS / chapter / f"{chapter}.ipynb"
    path = nested if nested.exists() else CHAPTERS / f"{chapter}.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    if chapter == "ch04":
        nb["cells"].append(
            {
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": [ROLLOUT_PROBE],
            },
        )
    t0 = time.time()
    try:
        failure = rn._execute(path, nb, "ollama-cloud", timeout, 12)  # noqa: SLF001
    except Exception as e:  # noqa: BLE001
        failure = f"runner: {type(e).__name__}: {e}"
    text = _outputs(nb)
    misses = (
        []
        if failure
        else [
            desc for desc, ok in EXPECTATIONS.get(chapter, []) if not ok(text)
        ]
    )
    return {
        "model": model,
        "chapter": chapter,
        "ran": failure is None,
        "behaved": failure is None and not misses,
        "failure": failure,
        "misses": misses,
        "seconds": round(time.time() - t0),
        "tools": dict(_tools(text)),
    }


def models() -> None:
    """List the Ollama Cloud models available to OLLAMA_API_KEY."""
    req = urllib.request.Request(
        "https://ollama.com/api/tags",
        headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        names = sorted(m["name"] for m in json.load(resp)["models"])
    print("\n".join(names))


def run(
    *model_names: str,
    chapters: str | tuple[str, ...] = "ch03,ch04,ch05,ch06,ch07",
    runs: int = 1,
    out: str = "benchmark.jsonl",
    timeout: float = 300,
    isolate_home: bool = True,
) -> None:
    """Benchmark each model on the chapters, appending to `out`.

    Args:
        *model_names (str): Ollama Cloud model names, as `models` lists.
        chapters (str | tuple[str, ...]): Chapter notebooks to run,
            comma-separated (fire parses `--chapters ch04,ch06` into a
            tuple). Defaults to the quick screening set, ch03-ch07.
        runs (int): Times to run each chapter per model. Defaults to 1.
        out (str): JSON-lines file to append results to.
        timeout (float): Per-cell timeout in seconds. Defaults to 300.
        isolate_home (bool): Run with an empty HOME so user-level skills
            stay out of the catalog. Defaults to True.
    """
    with _isolated_home() if isolate_home else contextlib.nullcontext():
        _run_all(model_names, chapters, runs, out, timeout)


def _run_all(
    model_names: tuple[str, ...],
    chapters: str | tuple[str, ...],
    runs: int,
    out: str,
    timeout: float,
) -> None:
    names = chapters.split(",") if isinstance(chapters, str) else chapters
    chosen = [str(c).strip() for c in names if str(c).strip()]
    with Path(out).open("a", encoding="utf-8") as f:
        for model in model_names:
            os.environ["OLLAMA_MODEL"] = model
            for _ in range(runs):
                for chapter in chosen:
                    rec = _run_once(model, chapter, timeout)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    why = rec["failure"] or "; ".join(rec["misses"])
                    mark = "ok" if rec["behaved"] else "X"
                    print(
                        f"{model:24s} {chapter} {mark} {rec['seconds']}s"
                        f" {why[:100]}",
                        flush=True,
                    )


def summary(path: str) -> None:
    """Rank the models in a results file: behaved first, then speed.

    Args:
        path (str): A JSON-lines file written by `run`.
    """
    recs = [json.loads(line) for line in Path(path).read_text().splitlines()]
    by_model: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in recs:
        by_model[rec["model"]].append(rec)

    def rank(item: tuple[str, list[dict[str, Any]]]) -> tuple[float, float]:
        rs = item[1]
        return (-sum(r["behaved"] for r in rs) / len(rs), _secs(rs))

    for model, rs in sorted(by_model.items(), key=rank):
        per: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for r in rs:
            per[r["chapter"]][0] += r["behaved"]
            per[r["chapter"]][1] += 1
        chapters = " ".join(f"{c}:{b}/{n}" for c, (b, n) in sorted(per.items()))
        issues = collections.Counter(
            f"{r['chapter']}: {(r['failure'] or '; '.join(r['misses']))[:60]}"
            for r in rs
            if not r["behaved"]
        )
        behaved = sum(r["behaved"] for r in rs)
        print(
            f"{model:24s} behaved {behaved}/{len(rs)}  {chapters}"
            f"  avg {_secs(rs):.0f}s/run",
        )
        for issue, n in issues.most_common():
            print(f"    {n}x {issue}")


def _secs(rs: list[dict[str, Any]]) -> float:
    return float(sum(r["seconds"] for r in rs)) / len(rs)


if __name__ == "__main__":
    fire.Fire({"models": models, "run": run, "summary": summary})
