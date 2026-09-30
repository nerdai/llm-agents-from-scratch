"""Run notebooks against a chosen LLM provider and record the result.

The runner behind `docs/notebook-status.md`. For each notebook it starts
a fresh kernel with the environment `make_llm()` needs for the requested
provider, executes every code cell, answers any human-input prompt with
a scripted reply, and writes one ledger entry per (notebook, provider)
into `notebook_status.yaml` at the repo root:

    examples/ch04.ipynb:
      openai: {status: pass, last_tested: '2026-09-29', model: gpt-5,
               commit: 1c0c22b0, seconds: 84}

Statuses:

    pass       ran to completion
    fail       a cell raised or timed out (the `note` says which)
    not-wired  the notebook constructs its LLM directly, so LLM_PROVIDER
               would have no effect; it is only executed for `ollama`.
               A notebook using `ollama_settings()` runs on both Ollama
               columns and is not-wired for OpenAI and Anthropic
    no-llm     nothing in the notebook builds an LLM; it is only executed
               for `ollama`, where a failure is still recorded as `fail`

Why not nbclient: it starts kernels with allow_stdin=False, so every
`input()`, and therefore every rich Prompt/Confirm the HITL notebooks
use, fails instantly. Driving the kernel over jupyter_client directly
lets us answer `input_request`s: "y" to an approval gate, a number to a
number question, a name to a name question.

A cell tagged `await:<name>` (for example `await:handler`) makes the
runner wait on that future before running the cell, standing in for the
pause a reader takes between starting background work and reading its
result.

Outputs are discarded by default so committed notebooks are never
overwritten by a verification run; pass --keep-outputs to write the
executed cells back (serialization mirrors the file, as
insert_provider_note.py does).

Usage:

    uv run python _scripts/run_notebook.py examples/ch04.ipynb --provider openai
    uv run python _scripts/run_notebook.py --all --provider anthropic
    uv run python _scripts/run_notebook.py --all --timeout 1500
"""

import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from queue import Empty
from typing import Any

import fire
import yaml
from jupyter_client import KernelManager

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "notebook_status.yaml"
PROVIDERS = ("ollama", "ollama-cloud", "openai", "anthropic")
NOTEBOOK_GLOBS = (
    "examples/ch*.ipynb",
    "more-examples/*/*.ipynb",
    "capstones/*/*.ipynb",
)
#: What make_llm() prints, mapped back to the provider key it means.
_USING = re.compile(
    r"✓ Using (Ollama Cloud|Ollama|OpenAI|Anthropic) \(([^)]+)\)",
)
_USING_KEY = {
    "Ollama": "ollama",
    "Ollama Cloud": "ollama-cloud",
    "OpenAI": "openai",
    "Anthropic": "anthropic",
}
_DIRECT = re.compile(r"(?:Ollama|OpenAI|Anthropic)LLM\(")
_DIRECT_MODEL = re.compile(
    r"(?:Ollama|OpenAI|Anthropic)LLM\(\s*model=\"([^\"]+)\"",
)
_PROVIDER_VARS = (
    "LLM_PROVIDER",
    "OLLAMA_API_KEY",
    "OLLAMA_HOST",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)
_POLL = 0.05
#: Cell tag prefix: wait on the named future before running the cell.
AWAIT_TAG = "await:"


def _tracked_notebooks() -> list[Path]:
    roots = sorted({g.split("/", maxsplit=1)[0] for g in NOTEBOOK_GLOBS})
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--", *roots],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return sorted(Path(p) for g in NOTEBOOK_GLOBS for p in glob.glob(g))
    return sorted(Path(p) for p in listed if p.endswith(".ipynb"))


def _head_commit() -> str:
    done = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )
    return done.stdout.strip() if done.returncode == 0 else ""


def _kernel_env(provider: str) -> dict[str, str]:
    """The environment a kernel needs for `make_llm()` to pick `provider`."""
    env = {k: v for k, v in os.environ.items() if k not in _PROVIDER_VARS}
    if provider == "ollama":
        return env
    if provider == "ollama-cloud":
        key = os.environ.get("OLLAMA_API_KEY")
        if not key:
            raise SystemExit("ollama-cloud needs OLLAMA_API_KEY set")
        env["OLLAMA_API_KEY"] = key
        return env
    var = f"{provider.upper()}_API_KEY"
    key = os.environ.get(var)
    if not key:
        raise SystemExit(f"{provider} needs {var} set")
    env[var] = key
    env["LLM_PROVIDER"] = provider
    return env


def _answer(recent_text: str, prompt: str) -> str:
    """The scripted human's reply to an input prompt."""
    text = f"{recent_text}\n{prompt}"
    if "Approve this result?" in text:
        return "y"
    if re.search(r"\bname\b", text, re.I):
        return "Andrei"
    if re.search(r"\b(number|integer|start|value)\b", text, re.I):
        return "6"
    return "yes"


@dataclass
class _CellRun:
    outputs: list[dict[str, Any]] = field(default_factory=list)
    tail: str = ""
    error: str | None = None
    idle: bool = False
    replied: bool = False


def _pump_stdin(kc: Any, msg_id: str, run: _CellRun) -> None:
    try:
        m = kc.stdin_channel.get_msg(timeout=_POLL)
    except Empty:
        return
    if m["header"]["msg_type"] != "input_request":
        return
    # the prompt text is printed before input() is called; make sure it
    # has been read from iopub before choosing the reply
    _pump_iopub(kc, msg_id, run)
    ans = _answer(run.tail[-1500:], m["content"].get("prompt", ""))
    run.outputs.append(
        {
            "output_type": "stream",
            "name": "stdout",
            "text": f"[scripted human] > {ans}\n",
        },
    )
    kc.input(ans)


def _pump_iopub(kc: Any, msg_id: str, run: _CellRun) -> None:
    try:
        while True:
            m = kc.get_iopub_msg(timeout=_POLL)
            if m["parent_header"].get("msg_id") != msg_id:
                continue
            kind, c = m["header"]["msg_type"], m["content"]
            if kind == "stream":
                run.tail += c["text"]
                last = run.outputs[-1] if run.outputs else None
                if (
                    last is not None
                    and last["output_type"] == "stream"
                    and last["name"] == c["name"]
                ):
                    last["text"] += c["text"]  # coalesce like Jupyter does
                else:
                    run.outputs.append(
                        {
                            "output_type": kind,
                            "name": c["name"],
                            "text": c["text"],
                        },
                    )
            elif kind in ("execute_result", "display_data"):
                # rich renders prompts through display() under ipykernel,
                # so the prompt text arrives here, not on the stream
                run.tail += c["data"].get("text/plain", "")
                out = {
                    "output_type": kind,
                    "data": c["data"],
                    "metadata": c.get("metadata", {}),
                }
                if kind == "execute_result":
                    out["execution_count"] = c.get("execution_count")
                run.outputs.append(out)
            elif kind == "error":
                run.error = f"{c['ename']}: {c['evalue']}"
                run.outputs.append(
                    {
                        "output_type": kind,
                        "ename": c["ename"],
                        "evalue": c["evalue"],
                        "traceback": c["traceback"],
                    },
                )
            elif kind == "status" and c["execution_state"] == "idle":
                run.idle = True
    except Empty:
        pass


def _pump_shell(kc: Any, msg_id: str, run: _CellRun) -> None:
    try:
        r = kc.shell_channel.get_msg(timeout=_POLL)
    except Empty:
        return
    if r["parent_header"].get("msg_id") == msg_id:
        run.replied = True


def _run_cell(kc: Any, cell: dict[str, Any], timeout: float) -> None:
    """Execute one cell in place, answering stdin prompts.

    Outputs are written into the cell as they arrive, so a failed cell
    keeps what it produced. Raises on error/timeout.
    """
    msg_id = kc.execute("".join(cell["source"]), allow_stdin=True)
    run = _CellRun()
    cell["outputs"] = run.outputs
    start = time.time()
    while not (run.replied and run.idle):
        if time.time() - start > timeout:
            raise TimeoutError(f"cell exceeded {timeout:.0f}s")
        _pump_stdin(kc, msg_id, run)
        _pump_iopub(kc, msg_id, run)
        _pump_shell(kc, msg_id, run)
    if run.error:
        raise RuntimeError(run.error)


def _code(nb: dict[str, Any]) -> str:
    return "".join(
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
    )


def _wiring(nb: dict[str, Any]) -> tuple[str, str]:
    """How the notebook gets its LLM.

    Returns (wiring, model hint), where wiring is one of `wired`
    (`make_llm()`: every provider), `ollama-wired` (`ollama_settings()`:
    local Ollama or Ollama Cloud only), `direct` or `none`.
    """
    source = _code(nb)
    if "make_llm(" in source:
        return "wired", ""
    if "ollama_settings(" in source:
        return "ollama-wired", ""
    if _DIRECT.search(source):
        m = _DIRECT_MODEL.search(source)
        return "direct", m.group(1) if m else "constructed directly"
    return "none", ""


def _stream_text(nb: dict[str, Any]) -> str:
    return "".join(
        o["text"]
        for c in nb["cells"]
        for o in c.get("outputs", [])
        if o.get("output_type") == "stream"
    )


def _progress(path: Path, index: int, t0: float, err: str = "") -> None:
    state = f"failed: {err}" if err else "ok"
    print(
        f"  {path.name} cell {index}: {state} ({time.time() - t0:.0f}s)",
        file=sys.stderr,
        flush=True,
    )


def _dump(path: Path, nb: dict[str, Any], provider: str) -> Path:
    """Write an executed notebook where a failure can be inspected."""
    out = Path(tempfile.gettempdir()) / "notebook_runs" / provider / path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


def _await_tagged(kc: Any, cell: dict[str, Any], timeout: float) -> None:
    """Stand in for a reader's pause before a cell tagged `await:<name>`.

    Some notebooks start work in one cell and read its result a few cells
    later, relying on the reader's pace for it to finish. The tag names the
    future to wait on (for example `await:handler`); the runner waits for
    it without raising, so the tagged cell still sees any exception the
    way it would interactively. Tags are hidden from readers.
    """
    for tag in cell.get("metadata", {}).get("tags", []):
        if not tag.startswith(AWAIT_TAG):
            continue
        name = tag.removeprefix(AWAIT_TAG)
        if not name.isidentifier():
            raise ValueError(f"bad tag {tag!r}: expected await:<name>")
        pause = {
            "source": [f"import asyncio as _nb\nawait _nb.wait([{name}])"],
        }
        _run_cell(kc, pause, timeout)


def _execute(
    path: Path,
    nb: dict[str, Any],
    provider: str,
    timeout: float,
) -> str | None:
    """Run every code cell in a fresh kernel; return the first failure."""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            # drop committed outputs so only this run's are classified
            cell["outputs"] = []
            cell["execution_count"] = None
    km = KernelManager(kernel_name="python3")
    km.start_kernel(cwd=str(path.parent), env=_kernel_env(provider))
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=60)
    count = 0
    try:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            count += 1
            t0 = time.time()
            try:
                _await_tagged(kc, cell, timeout)
                _run_cell(kc, cell, timeout)
                cell["execution_count"] = count
            except Exception as e:  # noqa: BLE001
                _progress(path, i, t0, str(e))
                return f"cell {i}: {e}"
            _progress(path, i, t0)
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    return None


def _entry(
    status: str,
    model: str = "",
    note: str = "",
    seconds: int = 0,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "status": status,
        "last_tested": date.today().isoformat(),
        "model": model,
        "commit": _head_commit(),
        "seconds": seconds,
    }
    if note:
        entry["note"] = note
    return entry


def _skip(wiring: str, provider: str) -> dict[str, Any] | None:
    """The entry for a run that would not exercise `provider`, if any."""
    if wiring == "none" and provider != "ollama":
        return _entry("no-llm")
    if wiring == "ollama-wired" and provider not in ("ollama", "ollama-cloud"):
        return _entry(
            "not-wired",
            note="Ollama only (ollama_settings); not run",
        )
    if wiring == "direct" and provider != "ollama":
        return _entry("not-wired", note="constructs its LLM directly; not run")
    return None


def run_one(
    path: Path,
    provider: str,
    timeout: float,
    keep_outputs: bool,
) -> dict[str, Any]:
    """Run one notebook against `provider` and return its ledger entry."""
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)
    wiring, hint = _wiring(nb)
    if skipped := _skip(wiring, provider):
        return skipped
    t0 = time.time()
    failure = _execute(path, nb, provider, timeout)
    seconds = round(time.time() - t0)
    used = _USING.search(_stream_text(nb))
    model = used.group(2) if used else hint
    if failure:
        print(
            f"  executed notebook: {_dump(path, nb, provider)}",
            file=sys.stderr,
        )
        return _entry("fail", model, failure, seconds)
    if wiring == "none":
        return _entry("no-llm", seconds=seconds)
    wired = wiring in ("wired", "ollama-wired")
    if wired and (not used or _USING_KEY[used[1]] != provider):
        ran_on = used[1] if used else "an unreported provider"
        return _entry("fail", model, f"make_llm() ran on {ran_on}", seconds)
    if keep_outputs:
        with path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=raw.isascii())
            f.write("\n")
    return _entry("pass", model, seconds=seconds)


def main(
    *paths: str,
    provider: str = "ollama",
    all: bool = False,  # noqa: A002 - mirrors the CLI flag
    timeout: float = 1200,
    keep_outputs: bool = False,
    ledger: str = str(LEDGER),
) -> None:
    """Run notebooks against `provider` and record results in the ledger.

    Args:
        *paths (str): Notebooks to run. Ignored when --all is given.
        provider (str): One of ollama, ollama-cloud, openai, anthropic.
            Defaults to "ollama" (local).
        all (bool): Run every tracked notebook. Defaults to False.
        timeout (float): Per-cell timeout in seconds. Defaults to 1200.
        keep_outputs (bool): Write executed outputs back into the notebook
            on a pass. Defaults to False (outputs discarded).
        ledger (str): Path of the YAML ledger to update.
    """
    if provider not in PROVIDERS:
        raise SystemExit(f"provider must be one of {', '.join(PROVIDERS)}")
    targets = _tracked_notebooks() if all else [Path(p) for p in paths]
    if not targets:
        raise SystemExit("nothing to run: pass notebook paths or --all")
    ledger_path = Path(ledger)
    data: dict[str, Any] = {}
    if ledger_path.exists():
        data = yaml.safe_load(ledger_path.read_text(encoding="utf-8")) or {}
    failed = 0
    for path in targets:
        key = str(path.relative_to(REPO)) if path.is_absolute() else str(path)
        entry = run_one(path, provider, timeout, keep_outputs)
        data.setdefault(key, {})[provider] = entry
        ledger_path.write_text(
            yaml.safe_dump(data, sort_keys=True),
            encoding="utf-8",
        )
        failed += entry["status"] == "fail"
        model = f" ({entry['model']})" if entry["model"] else ""
        note = f"  {entry['note']}" if entry.get("note") else ""
        print(
            f"{key}  {provider}: {entry['status']}{model}"
            f"  {entry['seconds']}s{note}",
            flush=True,
        )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    fire.Fire(main)
