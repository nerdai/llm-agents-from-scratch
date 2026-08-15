"""Wrap a raster PNG in an SVG with a fixed physical print size.

A screenshot can't become genuine vector art -- this doesn't attempt
that. It solves a narrower, real problem: placing a PNG at "actual
size" in a layout tool (Canva, InDesign) requires the file to already
carry a physical size, or every placement is a manual, inconsistent
resize (the same pain point `scripts/set_svg_print_size.py` fixes for
PlantUML diagrams). Wrapping the PNG in an SVG with explicit
`width`/`height` in inches gives it that fixed physical size directly,
without needing vector content.

Sizing: fits the image to the book's max figure box (default 5.6in x
7in), preserving aspect ratio -- capped by whichever dimension binds,
never upscaled past what the source pixel count comfortably supports
at print resolution (warns if the fitted size would drop below
300 DPI).

Deliberately dependency-free (stdlib only, argparse rather than
`fire`) so it can run via a bare `sys.executable` from any
environment -- e.g. `scripts/capture_inspector_screenshots.py`'s own
isolated PEP 723 env, which has `playwright` but not this project's
own dev dependencies.
"""

import argparse
import base64
import struct
from pathlib import Path

DEFAULT_MAX_WIDTH_IN = 5.6
DEFAULT_MAX_HEIGHT_IN = 7.0
MIN_PRINT_DPI = 300


def _png_dimensions(png_bytes: bytes) -> tuple[int, int]:
    # PNG: 8-byte signature, then an IHDR chunk whose first 8 bytes
    # (after the 4-byte length + "IHDR" type) are width/height as
    # big-endian uint32 -- no need for a Pillow dependency just to
    # read this.
    width, height = struct.unpack(">II", png_bytes[16:24])
    return width, height


def _wrap_one(
    png_path: Path,
    max_width_in: float,
    max_height_in: float,
) -> Path:
    png_bytes = png_path.read_bytes()
    px_width, px_height = _png_dimensions(png_bytes)
    aspect = px_height / px_width

    width_in = max_width_in
    height_in = width_in * aspect
    if height_in > max_height_in:
        height_in = max_height_in
        width_in = height_in / aspect

    dpi = px_width / width_in
    if dpi < MIN_PRINT_DPI:
        print(
            f"  warning: {png_path.name} fitted to "
            f"{width_in:.2f}in x {height_in:.2f}in is only "
            f"{dpi:.0f} DPI (below {MIN_PRINT_DPI})",
        )

    b64 = base64.b64encode(png_bytes).decode("ascii")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_in:.4f}in" height="{height_in:.4f}in" '
        f'viewBox="0 0 {px_width} {px_height}">'
        f'<image width="{px_width}" height="{px_height}" '
        f'href="data:image/png;base64,{b64}"/>'
        f"</svg>"
    )
    svg_path = png_path.with_suffix(".svg")
    svg_path.write_text(svg)
    return svg_path


def main(
    png_dir: Path,
    max_width_in: float = DEFAULT_MAX_WIDTH_IN,
    max_height_in: float = DEFAULT_MAX_HEIGHT_IN,
) -> None:
    """Wrap every PNG under png_dir in a fixed-size SVG alongside it.

    Args:
        png_dir (Path): Directory to search for `.png` files.
        max_width_in (float): Max figure width, in inches.
        max_height_in (float): Max figure height, in inches.

    Examples:
        >>> python scripts/wrap_png_as_svg.py --png_dir screenshots
    """
    wrapped = 0
    for png_path in sorted(png_dir.rglob("*.png")):
        svg_path = _wrap_one(png_path, max_width_in, max_height_in)
        print(f"  wrote {svg_path}")
        wrapped += 1
    print(f"Wrapped {wrapped} PNG file(s) as sized SVGs under {png_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--png_dir",
        type=Path,
        required=True,
        help="Directory to search for .png files.",
    )
    parser.add_argument(
        "--max_width_in",
        type=float,
        default=DEFAULT_MAX_WIDTH_IN,
        help="Max figure width, in inches.",
    )
    parser.add_argument(
        "--max_height_in",
        type=float,
        default=DEFAULT_MAX_HEIGHT_IN,
        help="Max figure height, in inches.",
    )
    args = parser.parse_args()
    main(args.png_dir, args.max_width_in, args.max_height_in)
