"""Render `notebook_status.yaml` as `docs/notebook-status.md`.

One table per notebook tier (chapter notebooks, additional examples,
capstones), one row per tracked notebook, one column per provider. A cell
shows the last recorded run for that (notebook, provider) pair:

    ✅ `gpt-5`                 passed, with that model
    ❌                         failed (the note goes in the footnotes)
    ⚪ not wired               the notebook constructs its LLM directly,
                               so LLM_PROVIDER would have no effect
    ∅ no LLM                   nothing in the notebook builds an LLM
    –                          never run on that provider

A single "Last tested" date heads the page: the most recent run in the
ledger, plus the oldest one when they differ, so a partial re-run never
passes off stale results as fresh.

The page is generated: edit the ledger (or re-run
`_scripts/run_notebook.py`) and re-render, never the markdown.

Usage:

    uv run python _scripts/render_notebook_status.py
"""

import glob
import subprocess
from pathlib import Path
from typing import Any

import fire
import yaml

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "notebook_status.yaml"
PAGE = REPO / "docs" / "notebook-status.md"
#: Column order: the recommended provider first. Local Ollama is left out
#: on purpose: its results say more about the machine than the notebook.
PROVIDERS = (
    ("ollama-cloud", "Ollama Cloud"),
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
)
TIERS = (
    ("examples/", "Chapter notebooks"),
    ("more-examples/", "Additional examples"),
    ("capstones/", "Capstones"),
)
NOTEBOOK_GLOBS = (
    "examples/ch*.ipynb",
    "more-examples/*/*.ipynb",
    "capstones/*/*.ipynb",
)


def _tracked_notebooks() -> list[str]:
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
        listed = [p for g in NOTEBOOK_GLOBS for p in glob.glob(g)]
    return sorted(p for p in listed if p.endswith(".ipynb"))


def _docs_link(path: str) -> str:
    """Markdown link to the rendered notebook when docs/ carries it."""
    name = path.rsplit("/", maxsplit=1)[-1]
    docs_rel = (
        path.replace("examples/", "notebooks/", 1)
        if path.startswith("examples/")
        else path
    )
    if (REPO / "docs" / docs_rel).exists():
        return f"[`{name}`]({docs_rel})"
    return f"`{name}`"


_PLAIN = {"not-wired": "⚪ not wired", "no-llm": "∅ no LLM"}


def _cell(entry: dict[str, Any] | None, notes: list[str], label: str) -> str:
    if entry is None:
        return "–"
    status = str(entry.get("status"))
    model = f" `{entry['model']}`" if entry.get("model") else ""
    if status == "pass":
        if caveat := entry.get("caveat"):
            notes.append(f"{label}: {caveat}")
            return f"✅{model}[^{len(notes)}]"
        return f"✅{model}"
    if status == "fail":
        notes.append(f"{label}: {entry.get('note', 'failed')}")
        return f"❌[^{len(notes)}]"
    return _PLAIN.get(status, f"? {status}")


def _last_tested(ledger: dict[str, Any]) -> str:
    """One date for the page: the newest run, and the oldest if older."""
    dates = sorted(
        str(entry["last_tested"])
        for runs in ledger.values()
        for entry in runs.values()
        if entry.get("last_tested")
    )
    if not dates:
        return "never"
    if dates[0] == dates[-1]:
        return dates[-1]
    return f"{dates[-1]} (oldest result: {dates[0]})"


def render(ledger: dict[str, Any]) -> str:
    """Render the whole status page from the ledger mapping."""
    lines = [
        "# Notebook Status",
        "",
        "Which notebooks have been run end to end against which LLM",
        "provider. Generated from `notebook_status.yaml` by",
        "`_scripts/render_notebook_status.py`; runs are recorded by",
        "`_scripts/run_notebook.py`. Edit the ledger and re-render, not",
        "this page.",
        "",
        f"**Last tested:** {_last_tested(ledger)}",
        "",
        "| Symbol | Meaning |",
        "|---|---|",
        "| ✅ `model` | Ran to completion with that model |",
        "| ✅ `model` + footnote | Passed with a caveat; see the footnote |",
        "| ❌ | Failed; see the footnote |",
        "| ⚪ not wired | The notebook constructs its LLM directly, so"
        " the provider setting would have no effect; not run |",
        "| ∅ no LLM | Nothing in the notebook builds an LLM;"
        " run under Ollama Cloud only |",
        "| – | Never run on that provider |",
        "",
        "Human-in-the-loop prompts are answered by a scripted reply during",
        "these runs, so a pass means the notebook runs to completion, not",
        "that its prompts read well.",
        "",
        "A cell tagged `await:<name>` (ch04's result cell, for example) is",
        "run after the runner waits on that future, standing in for the",
        "pause a reader takes between starting a task and reading its",
        "result. Those notebooks are meant to be run cell by cell, not with",
        "Run All.",
        "",
    ]
    tracked = _tracked_notebooks()
    for prefix, title in TIERS:
        rows = [p for p in tracked if p.startswith(prefix)]
        if not rows:
            continue
        notes: list[str] = []
        lines += [f"## {title}", ""]
        # wrapper lets extra.css keep cells on one line, so a wide table
        # scrolls sideways instead of wrapping into tall, narrow cells
        lines += ['<div class="notebook-status" markdown>', ""]
        lines.append(
            "| Notebook | " + " | ".join(t for _, t in PROVIDERS) + " |",
        )
        lines.append("|---|" + "---|" * len(PROVIDERS))
        for path in rows:
            runs = ledger.get(path, {})
            cells = [
                _cell(runs.get(key), notes, f"`{path}` / {title_}")
                for key, title_ in PROVIDERS
            ]
            lines.append(f"| {_docs_link(path)} | " + " | ".join(cells) + " |")
        lines += ["", "</div>", ""]
        # footnote numbering restarts per table, so scope the notes to it
        for n, note in enumerate(notes, start=1):
            lines.append(f"[^{n}]: {note}")
        if notes:
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(ledger: str = str(LEDGER), page: str = str(PAGE)) -> None:
    """Render the ledger to the docs page.

    Args:
        ledger (str): Path of the YAML ledger. Defaults to the repo root one.
        page (str): Output markdown path. Defaults to docs/notebook-status.md.
    """
    data = yaml.safe_load(Path(ledger).read_text(encoding="utf-8")) or {}
    Path(page).write_text(render(data), encoding="utf-8")
    print(f"wrote {page} ({sum(len(v) for v in data.values())} runs)")


if __name__ == "__main__":
    fire.Fire(main)
