# /// script
# dependencies = ["playwright>=1.55"]
# ///
"""Capture high-resolution Agent Inspector screenshots for the book.

Replaces the low-resolution, manually-captured screenshots used in
the Ch08 manuscript (Figure 8.9 and friends) per Ross Turner's review
feedback. Drives a real Agent Inspector session -- against a live
Ollama backend, not a scripted double -- through Playwright, so the
captured reasoning text is authentic model output, not canned.

Selector conventions (role/text-based, e.g. `get_by_role("button",
name="Approve")`) mirror the Agent Inspector repo's own Playwright E2E
suite (`frontend/e2e/helpers.ts`), which proved this UI has no
`data-testid` attributes but is fully drivable via accessible roles.

`playwright` is declared via the PEP 723 block above rather than added
to this repo's `pyproject.toml` -- it's a one-off capture-tooling need,
not a real project dependency. First run:

    uv run scripts/capture_inspector_screenshots.py --help
    playwright install chromium   # once, if not already installed

Then, with `ollama serve` running and `qwen3:14b` pulled:

    uv run scripts/capture_inspector_screenshots.py

This launches `examples/demo.py` via the Agent Inspector CLI itself
(installed on demand via `uvx`, kept out of this repo's dependency
tree -- see `examples/demo.py`'s own docstring), so no separate
backend needs to be running first.
"""

import argparse
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AGENT_SCRIPT = REPO_ROOT / "examples" / "demo.py"
# The latest PyPI release (0.1.3) 500s on session creation --
# `create_session` still calls `session.handler.skills.items()`, but
# TaskHandler's attribute is `skills_registry` (renamed pre-0.1.3).
# Already fixed on the inspector repo's `main` branch, just not
# released yet -- installing from git unblocks capture in the
# meantime. Swap back to "llm-agents-from-scratch-inspector" once a
# fixed version ships.
DEFAULT_INSPECTOR_SOURCE = (
    "git+https://github.com/nerdai/llm-agents-from-scratch-inspector"
)
# Manuscript figures live in each user's own local `figures_path` (see
# `scripts/prepare_book_figures.py` / `book_figures.yml`), never
# tracked in this repo -- so the default here is a untracked scratch
# directory, not a docs/assets path. Point --output-dir at your real
# figures_path/ch08 when capturing for real.
DEFAULT_OUTPUT_DIR = REPO_ROOT / "screenshots"
DEFAULT_PORT = 8123
VIEWPORT = {"width": 1440, "height": 900}
DEVICE_SCALE_FACTOR = 2  # crisp @2x captures for print
SERVER_STARTUP_TIMEOUT_SECONDS = 60
STEP_POLL_INTERVAL_SECONDS = 0.5


def _wait_for_server(url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)  # noqa: S310
            return
        except (urllib.error.URLError, ConnectionError):
            time.sleep(0.5)
    raise TimeoutError(f"Server at {url} did not become ready in time.")


_PHASE_BADGE_RE = re.compile(
    r"^(Awaiting|Task complete|Aborted|Calling backend)",
)


def _phase_badge(page: Page):
    # Python Playwright's get_by_text treats a plain str as a literal
    # substring match (unlike the TS API's implicit regex support), so
    # this needs an actual compiled pattern to behave like helpers.ts's
    # `getByText(/^(Awaiting|...)/)`.
    return page.locator("header").get_by_text(_PHASE_BADGE_RE)


def _click_and_wait(page: Page, button_name: str) -> None:
    page.get_by_role("button", name=button_name).click()
    _phase_badge(page).wait_for(state="visible")
    while "Calling backend" in (_phase_badge(page).text_content() or ""):
        time.sleep(STEP_POLL_INTERVAL_SECONDS)


def _wait_for_session_created(page: Page, timeout_seconds: float = 30) -> None:
    # The E2E suite's helpers.ts polls the URL's `?session=` query
    # param, but this build never syncs it there (confirmed live: the
    # session sidebar and phase badge both update correctly, the URL
    # just stays at "/") -- so wait on the phase badge instead, which
    # reliably flips to "Awaiting get_next_step()" once the session
    # exists.
    _phase_badge(page).wait_for(state="visible", timeout=timeout_seconds * 1000)


def _capture(page: Page, output_dir: Path, name: str) -> None:
    path = output_dir / f"{name}.png"
    page.screenshot(path=path)
    print(f"  wrote {path}")


def run(
    agent_script: Path,
    output_dir: Path,
    port: int,
    inspector_source: str,
) -> None:
    """Launch the Inspector against agent_script and capture screenshots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"

    print(f"Launching Agent Inspector on {base_url} ...")
    server = subprocess.Popen(  # noqa: S603
        [
            "uvx",
            "--from",
            inspector_source,
            "agent-inspector",
            "launch",
            str(agent_script),
            "--no-open",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_server(base_url, SERVER_STARTUP_TIMEOUT_SECONDS)
        print("Server ready.")

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(
                viewport=VIEWPORT,
                device_scale_factor=DEVICE_SCALE_FACTOR,
            )

            page.goto(base_url)
            page.locator("#task-input").wait_for(state="visible")
            _capture(page, output_dir, "agent-inspector-landing-page")

            page.get_by_role("button", name="Create session").click()
            _wait_for_session_created(page)

            # First get_next_step() -> run_step(): capture the
            # transient "in flight" state right after clicking, before
            # the real Ollama call resolves (mirrors the manuscript's
            # Figure 8.9).
            _click_and_wait(page, "get_next_step()")
            page.get_by_role("button", name="run_step(step)").click()
            time.sleep(STEP_POLL_INTERVAL_SECONDS)
            _capture(page, output_dir, "agent-inspector-step-in-flight")
            while "Calling backend" in (
                _phase_badge(page).text_content() or ""
            ):
                time.sleep(STEP_POLL_INTERVAL_SECONDS)

            # Drive the rest of the loop to the approval gate.
            while "Awaiting approval" not in (
                _phase_badge(page).text_content() or ""
            ):
                phase = _phase_badge(page).text_content() or ""
                if "run_step" in phase or "Awaiting run_step" in phase:
                    _click_and_wait(page, "run_step(step)")
                else:
                    _click_and_wait(page, "get_next_step()")

            _capture(page, output_dir, "agent-inspector-acceptance-gate")

            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    # Print-ready SVG alongside each PNG (fixed physical size baked
    # in, see that script's own docstring for why). A separate,
    # dependency-free script rather than inlined here so it stays
    # reusable for other non-screenshot PNGs later -- run via
    # sys.executable directly, not `uv run`, since it needs no
    # PEP 723 deps of its own.
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "wrap_png_as_svg.py"),
            "--png_dir",
            str(output_dir),
        ],
        check=True,
    )


def main() -> None:
    """Parse CLI args and run the capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent-script",
        type=Path,
        default=DEFAULT_AGENT_SCRIPT,
        help="Path to the agent_builder script (default: examples/demo.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write screenshots to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port to run the Inspector backend on.",
    )
    parser.add_argument(
        "--inspector-source",
        default=DEFAULT_INSPECTOR_SOURCE,
        help=(
            "uvx --from source for the Inspector CLI (default: git main, "
            "since the latest PyPI release 500s on session creation -- "
            "see the constant's comment above). Pass "
            "'llm-agents-from-scratch-inspector' once a fixed version "
            "ships to PyPI."
        ),
    )
    args = parser.parse_args()
    run(args.agent_script, args.output_dir, args.port, args.inspector_source)


if __name__ == "__main__":
    sys.exit(main())
