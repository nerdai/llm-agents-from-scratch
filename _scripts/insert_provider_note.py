r"""Insert or refresh the standard provider-setup note in every notebook.

Every notebook under `examples/`, `more-examples/` and `capstones/` gets
one identical markdown cell explaining that Ollama is the default and how
to opt into OpenAI or Anthropic via `make_llm()`. The wording lives here,
in `NOTE`, so changing it is a one-line edit plus a re-run rather than
~40 hand-edited notebooks that drift.

The cell is wrapped in HTML comment markers (`<!-- provider-note:start
-->` / `<!-- provider-note:end -->`), which render invisibly in Jupyter
and mkdocs. Re-runs find the cell by its start marker and rewrite it in
place, so the script is idempotent: running it twice is a no-op, and
running it after editing `NOTE` updates every notebook.

Placement: directly above the first code cell that runs `pip install`,
which every real notebook opens with. A notebook without one (the Part 3
capstone stubs) gets the note right after its title cell.

Serialization mirrors what each file already uses -- `json.dump(indent=1)`
with cell `source` as a list of lines, and non-ASCII kept as raw UTF-8 or
as `\uXXXX` escapes to match the file -- so a run that changes nothing
produces no diff, and a run that inserts one cell diffs as exactly that
cell.

Usage:

    uv run python _scripts/insert_provider_note.py            # apply
    uv run python _scripts/insert_provider_note.py --check    # dry run
"""

import glob
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import fire

START_MARKER = "<!-- provider-note:start -->"
END_MARKER = "<!-- provider-note:end -->"

NOTE_BODY = """## Running the LLM

These notebooks run on **Ollama by default**, the setup the book teaches.
If you do nothing, nothing changes: `make_llm()` starts a local Ollama
service when one isn't already running.

To use OpenAI or Anthropic instead:

1. Install the extra: `uv sync --extra openai` or `uv sync --extra anthropic`.
2. Export `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` before launching Jupyter.
3. Set `LLM_PROVIDER=openai` (or `anthropic`), or pass `provider="openai"`
   to `make_llm()`. A key on its own never switches providers, so one
   exported for unrelated work cannot reroute you off the Ollama path.

The switch applies wherever a notebook builds its LLM with `make_llm()`.
A notebook that constructs `OllamaLLM` directly stays on Ollama regardless
of these settings.

If you opted in but forgot to export the key, you will be prompted for it.
That is the safe path on hosted kernels, and it keeps the key out of the
saved notebook. Setting `OLLAMA_API_KEY` alone routes Ollama to Ollama
Cloud.

**Caveat:** the examples are tuned for qwen3. Output on gpt-5 or Claude
will differ from what is printed in the book, and prompt-sensitive
examples (the ch09 evaluator pattern, ch08 supervised trajectories) may
behave noticeably differently."""

NOTE = f"{START_MARKER}\n{NOTE_BODY}\n{END_MARKER}"

NOTEBOOK_GLOBS = (
    "examples/ch*.ipynb",
    "more-examples/*/*.ipynb",
    "capstones/*/*.ipynb",
)

#: Bespoke per-notebook cells the standard note replaces, matched on their
#: first line. Removed when the note is inserted or updated.
SUPERSEDED_HEADINGS = ("## Setting the backbone LLM of your agent",)


def _source(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _lines(text: str) -> list[str]:
    """Split into the list-of-lines shape nbformat writes `source` in."""
    return text.splitlines(keepends=True)


def _is_note(cell: dict[str, Any]) -> bool:
    if cell["cell_type"] != "markdown":
        return False
    return _source(cell).lstrip().startswith(START_MARKER)


def _is_superseded(cell: dict[str, Any]) -> bool:
    first_line = _source(cell).lstrip().splitlines()[:1]
    return (
        cell["cell_type"] == "markdown"
        and bool(first_line)
        and (first_line[0].strip() in SUPERSEDED_HEADINGS)
    )


def _anchor_index(cells: list[dict[str, Any]]) -> int:
    """Index to insert the note at: above the pip-install cell if present."""
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code" and "pip install" in _source(cell):
            return i
    return 1 if cells and cells[0]["cell_type"] == "markdown" else 0


def _write(path: Path, nb: dict[str, Any], *, ensure_ascii: bool) -> None:
    r"""Write `nb` back in the same shape the file already had.

    Notebooks saved by Jupyter hold non-ASCII as raw UTF-8; ones written
    by `json.dump` hold `\uXXXX` escapes. Mirroring whichever the file
    used keeps an insertion diffing as one cell, not every emoji.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=ensure_ascii)
        f.write("\n")
    try:
        import nbformat  # noqa: PLC0415
    except ImportError:  # pragma: no cover - validation is best-effort
        return
    nbformat.validate(nbformat.read(str(path), as_version=4))


def process(path: Path, check: bool = False) -> str:
    """Insert or refresh the note in one notebook.

    Returns:
        str: "unchanged", "updated", or "inserted", describing what was (or
            in check mode, would be) done.
    """
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)
    cells: list[dict[str, Any]] = nb["cells"]

    existing = [i for i, cell in enumerate(cells) if _is_note(cell)]
    superseded = [i for i, cell in enumerate(cells) if _is_superseded(cell)]

    if (
        len(existing) == 1
        and _source(cells[existing[0]]) == NOTE
        and not superseded
    ):
        return "unchanged"

    # normalize to exactly one note: drop superseded bespoke cells and any
    # duplicate copies beyond the first, then refresh or insert
    for i in sorted(set(superseded) | set(existing[1:]), reverse=True):
        del cells[i]
    existing = [i for i, cell in enumerate(cells) if _is_note(cell)]

    if existing:
        cells[existing[0]]["source"] = _lines(NOTE)
        action = "updated"
    else:
        # keys in nbformat's sorted order, so the cell reads like its neighbours
        cell: dict[str, Any] = {"cell_type": "markdown"}
        if nb.get("nbformat_minor", 0) >= 5:  # noqa: PLR2004 - ids arrived in 4.5
            cell["id"] = uuid.uuid4().hex[:8]
        cell["metadata"] = {}
        cell["source"] = _lines(NOTE)
        cells.insert(_anchor_index(cells), cell)
        action = "inserted"

    if not check:
        _write(path, nb, ensure_ascii=raw.isascii())
    return action


def _tracked_notebooks() -> list[Path]:
    """Every git-tracked notebook under the three notebook roots.

    Going through git rather than the filesystem skips gitignored scratch
    copies and `.ipynb_checkpoints/`, which the globs would otherwise pick
    up. Falls back to the globs when git is unavailable.
    """
    roots = sorted({g.split("/", maxsplit=1)[0] for g in NOTEBOOK_GLOBS})
    try:
        listed = subprocess.run(
            ["git", "ls-files", "--", *roots],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return sorted(Path(p) for g in NOTEBOOK_GLOBS for p in glob.glob(g))
    return sorted(Path(p) for p in listed if p.endswith(".ipynb"))


def main(*paths: str, check: bool = False) -> None:
    """Apply the note to every notebook, or report what a run would change.

    Args:
        *paths (str): Notebooks to process instead of every tracked one.
        check (bool): Dry run. Report per-notebook actions without writing,
            and exit non-zero if anything would change. Keyword-only, so a
            positional path is never mistaken for it. Defaults to False.
    """
    targets = [Path(p) for p in paths] if paths else _tracked_notebooks()
    counts: dict[str, int] = {}
    for path in targets:
        action = process(path, check=check)
        counts[action] = counts.get(action, 0) + 1
        print(f"{action:>9}  {path}")
    summary = ", ".join(f"{n} {k}" for k, n in sorted(counts.items()))
    print(f"\n{len(targets)} notebooks: {summary}")
    if check and (counts.get("inserted") or counts.get("updated")):
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(main)
