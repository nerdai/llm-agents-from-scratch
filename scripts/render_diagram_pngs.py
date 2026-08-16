# /// script
# dependencies = ["playwright>=1.55"]
# ///
"""Render a PNG alongside every rendered UML diagram SVG.

`make diagrams` produces `uml/rendered/**/*.svg` with a fixed physical
print size baked in (see `scripts/set_svg_print_size.py`). This adds
a matching PNG next to each one, at a DPI that reproduces that exact
physical size crisply in print (default 300 DPI).

Renders through headless Chromium (Playwright) rather than `cairosvg`:
`cairosvg` ignores PlantUML's `textLength`/`lengthAdjust="spacing"`
(used to force exact text width so labels fit their boxes regardless
of actual font metrics), rendering glyphs at their natural width
instead and overflowing box boundaries -- confirmed with a minimal
repro, text spilling out of every class-diagram box. Browsers
implement this correctly, and Chromium is already relied on elsewhere
in this repo (`scripts/capture_inspector_screenshots.py`) for
faithful SVG/UI rendering, so this reuses the same engine instead of
a second, less-correct one.

    uv run scripts/render_diagram_pngs.py --rendered_dir uml/rendered
    playwright install chromium   # once, if not already installed
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

DEFAULT_DPI = 300
CSS_PX_PER_INCH = 96  # browsers treat "in" units as fixed 96 CSS px/in


def main(rendered_dir: Path, dpi: int) -> None:
    """Render a PNG next to every SVG under rendered_dir.

    Args:
        rendered_dir (Path): Directory containing rendered SVGs.
        dpi (int): Render resolution, in dots per inch.
    """
    svg_paths = sorted(rendered_dir.rglob("*.svg"))
    device_scale_factor = dpi / CSS_PX_PER_INCH

    rendered = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for svg_path in svg_paths:
            page = browser.new_page(device_scale_factor=device_scale_factor)
            page.goto(svg_path.resolve().as_uri())
            png_path = svg_path.with_suffix(".png")
            page.locator("svg").screenshot(path=str(png_path))
            page.close()
            print(f"  wrote {png_path}")
            rendered += 1
        browser.close()
    print(f"Rendered {rendered} PNG file(s) under {rendered_dir} @ {dpi} DPI.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rendered_dir",
        type=Path,
        required=True,
        help="Directory containing rendered SVGs (searched recursively).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Render resolution, in dots per inch.",
    )
    args = parser.parse_args()
    main(args.rendered_dir, args.dpi)
