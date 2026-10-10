"""Merge per-provider ledgers from parallel runs into the main ledger.

The notebook-status workflow runs one job per provider, each starting
from the same `notebook_status.yaml` and updating only its own provider's
column. This takes that column from each job's copy and writes it into
the main ledger, leaving every other column as it was.

Usage:

    uv run python _scripts/merge_notebook_status.py \
        ollama-cloud=runs/ollama-cloud/notebook_status.yaml \
        openai=runs/openai/notebook_status.yaml
"""

import sys
from pathlib import Path
from typing import Any

import fire
import yaml

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "notebook_status.yaml"
PROVIDERS = ("ollama-cloud", "openai", "anthropic")


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def merge(
    base: dict[str, Any],
    provider: str,
    partial: dict[str, Any],
) -> int:
    """Copy `provider`'s entries from `partial` into `base`.

    Args:
        base (dict[str, Any]): The ledger to update in place.
        provider (str): The provider column to take from `partial`.
        partial (dict[str, Any]): A ledger written by one provider's run.

    Returns:
        int: How many entries changed.
    """
    changed = 0
    for notebook, runs in partial.items():
        entry = (runs or {}).get(provider)
        if entry is None:
            continue
        if base.get(notebook, {}).get(provider) != entry:
            base.setdefault(notebook, {})[provider] = entry
            changed += 1
    return changed


def main(*pairs: str, ledger: str = str(LEDGER)) -> None:
    """Merge `provider=path` ledgers into the main ledger.

    Args:
        *pairs (str): `provider=path` pairs, one per provider job. A path
            that doesn't exist (a job that never uploaded) is skipped.
        ledger (str): The ledger to update. Defaults to the repo root one.
    """
    ledger_path = Path(ledger)
    base = _load(ledger_path) if ledger_path.exists() else {}
    total = 0
    for pair in pairs:
        provider, sep, path = pair.partition("=")
        if not sep or provider not in PROVIDERS:
            raise SystemExit(f"expected provider=path, got {pair!r}")
        if not Path(path).exists():
            print(f"{provider}: no ledger at {path}, skipped", file=sys.stderr)
            continue
        n = merge(base, provider, _load(Path(path)))
        print(f"{provider}: {n} entries changed")
        total += n
    ledger_path.write_text(
        yaml.safe_dump(base, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {ledger_path} ({total} entries changed)")


if __name__ == "__main__":
    fire.Fire(main)
