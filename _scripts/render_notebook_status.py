"""Render `notebook_status.yaml` as `docs/notebook-status.md`.

One table per notebook tier (chapter notebooks, additional examples,
capstones), one row per tracked notebook, one column per provider. A cell
shows the last recorded run for that (notebook, provider) pair:

    ✅ 2026-09-29 `gpt-5`      passed on that date, with that model
    ❌ 2026-09-29              failed (the note goes in the footnotes)
    ⚪ not wired               the notebook constructs its LLM directly,
                               so LLM_PROVIDER would have no effect
    ∅ no LLM                   nothing in the notebook builds an LLM
    –                          never run on that provider

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
#: Column order: the recommended provider first, local Ollama last.
PROVIDERS = (
    ("ollama-cloud", "Ollama Cloud"),
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("ollama", "Ollama (local)"),
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


def _cell(entry: dict[str, Any] | None, notes: list[str], label: str) -> str:
    if entry is None:
        return "–"
    status = entry.get("status")
    when = entry.get("last_tested", "")
    model = f" `{entry['model']}`" if entry.get("model") else ""
    if status == "pass":
        return f"✅ {when}{model}"
    if status == "fail":
        notes.append(f"{label}: {entry.get('note', 'failed')} ({when})")
        return f"❌ {when}[^{len(notes)}]"
    if status == "not-wired":
        return "⚪ not wired"
    if status == "no-llm":
        return "∅ no LLM"
    return f"? {status}"


def render(ledger: dict[str, Any]) -> str:
    """Render the whole status page from the ledger mapping."""
    lines = [
        "# Notebook Status",
        "",
        "Which notebooks have been run end to end against which LLM",
        "provider, and when. Generated from `notebook_status.yaml` by",
        "`_scripts/render_notebook_status.py`; runs are recorded by",
        "`_scripts/run_notebook.py`. Edit the ledger and re-render, not",
        "this page.",
        "",
        "| Symbol | Meaning |",
        "|---|---|",
        "| ✅ date `model` | Ran to completion on that date with that model |",
        "| ❌ date | Failed; see the footnote |",
        "| ⚪ not wired | The notebook constructs its LLM directly, so"
        " `LLM_PROVIDER` would have no effect; only run on local Ollama |",
        "| ∅ no LLM | Nothing in the notebook builds an LLM;"
        " only run on local Ollama |",
        "| – | Never run on that provider |",
        "",
        "Human-in-the-loop prompts are answered by a scripted reply during",
        "these runs, so a pass means the notebook runs to completion, not",
        "that its prompts read well.",
        "",
    ]
    tracked = _tracked_notebooks()
    for prefix, title in TIERS:
        rows = [p for p in tracked if p.startswith(prefix)]
        if not rows:
            continue
        notes: list[str] = []
        lines += [f"## {title}", ""]
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
        lines.append("")
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
