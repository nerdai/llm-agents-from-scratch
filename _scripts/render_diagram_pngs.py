# /// script
# dependencies = ["playwright>=1.55"]
# ///
"""Render a PNG alongside every rendered UML diagram SVG.

`make diagrams` produces `uml/rendered/**/*.svg` with a fixed physical
print size baked in (see `_scripts/set_svg_print_size.py`). This adds
a matching PNG next to each one, at a DPI that reproduces that exact
physical size crisply in print (default 300 DPI).

Renders through headless Chromium (Playwright) rather than `cairosvg`:
`cairosvg` ignores PlantUML's `textLength`/`lengthAdjust="spacing"`
(used to force exact text width so labels fit their boxes regardless
of actual font metrics), rendering glyphs at their natural width
instead and overflowing box boundaries -- confirmed with a minimal
repro, text spilling out of every class-diagram box. Browsers
implement this correctly, and Chromium is already relied on elsewhere
in this repo (`_scripts/capture_inspector_screenshots.py`) for
faithful SVG/UI rendering, so this reuses the same engine instead of
a second, less-correct one.

Also injects a PNG `pHYs` chunk declaring the render DPI. Playwright's
screenshot doesn't write one, so the file's *pixel count* is the only
size information a PNG-consuming tool (Word, etc.) has -- diagrams
with different pixel counts (different natural size/complexity) then
get auto-fit to a page differently, defeating the whole point of
set_svg_print_size.py's uniform physical sizing: a diagram rendered at
fewer total pixels looks *larger* once placed, not smaller, because
it needs less shrinking to fit the same page width. Confirmed via a
minimal repro (no `pHYs` chunk in the raw Playwright output). The
`pHYs` chunk lets DPI-aware tools place the image at its true physical
size instead of guessing from pixel count.

    uv run _scripts/render_diagram_pngs.py --rendered_dir uml/rendered
    playwright install chromium   # once, if not already installed
"""

import argparse
import struct
import zlib
from pathlib import Path

from playwright.sync_api import sync_playwright

DEFAULT_DPI = 300
CSS_PX_PER_INCH = 96  # browsers treat "in" units as fixed 96 CSS px/in
METERS_PER_INCH = 0.0254
PNG_SIGNATURE_LEN = 8
IHDR_CHUNK_LEN = 25  # 4 (length) + 4 (type) + 13 (data) + 4 (CRC)


def _phys_chunk(dpi: float) -> bytes:
    pixels_per_meter = round(dpi / METERS_PER_INCH)
    # unit=1 -> pixels-per-meter (the only PNG spec option for a real
    # physical unit; 0 means "unspecified aspect ratio only").
    data = struct.pack(">IIB", pixels_per_meter, pixels_per_meter, 1)
    chunk_type = b"pHYs"
    crc = zlib.crc32(chunk_type + data)
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", crc)
    )


def _add_dpi_metadata(png_path: Path, dpi: float) -> None:
    png_bytes = png_path.read_bytes()
    # pHYs must appear before the first IDAT chunk; right after IHDR
    # (always the first chunk) is the conventional, always-valid spot.
    insert_at = PNG_SIGNATURE_LEN + IHDR_CHUNK_LEN
    fixed = png_bytes[:insert_at] + _phys_chunk(dpi) + png_bytes[insert_at:]
    png_path.write_bytes(fixed)


def main(rendered_dir: Path, dpi: int) -> None:
    """Render a PNG next to every SVG under rendered_dir.

    Args:
        rendered_dir (Path): Directory containing rendered SVGs.
        dpi (int): Render resolution, in dots per inch.

    Raises:
        ValueError: If rendered_dir doesn't exist/isn't a directory,
            no SVGs are found under it, or dpi isn't positive -- each
            would otherwise fail silently (0 files rendered) or
            produce a confusing failure later.
    """
    if not rendered_dir.is_dir():
        raise ValueError(f"rendered_dir does not exist: {rendered_dir}")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    svg_paths = sorted(rendered_dir.rglob("*.svg"))
    if not svg_paths:
        raise ValueError(f"No .svg files found under {rendered_dir}")
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
            _add_dpi_metadata(png_path, dpi)
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
