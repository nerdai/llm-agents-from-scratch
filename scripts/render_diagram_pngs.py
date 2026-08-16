# /// script
# dependencies = ["cairosvg>=2.7"]
# ///
"""Render a PNG alongside every rendered UML diagram SVG.

`make diagrams` produces `uml/rendered/**/*.svg` with a fixed physical
print size baked in (see `scripts/set_svg_print_size.py`). This adds
a matching PNG next to each one, at a DPI that reproduces that exact
physical size crisply in print (default 300 DPI).

Kept as a separate, self-contained script (PEP 723 metadata above)
rather than folded into `make diagrams` or added to this repo's
`pyproject.toml`: `cairosvg` needs the system `libcairo2` library,
which isn't guaranteed to be present on every machine/CI runner, and
PNG export isn't needed for every `make diagrams` run.

    uv run scripts/render_diagram_pngs.py --rendered_dir uml/rendered
"""

import argparse
from pathlib import Path

import cairosvg

DEFAULT_DPI = 300


def main(rendered_dir: Path, dpi: int) -> None:
    """Render a PNG next to every SVG under rendered_dir.

    Args:
        rendered_dir (Path): Directory containing rendered SVGs.
        dpi (int): Render resolution, in dots per inch.
    """
    rendered = 0
    for svg_path in sorted(rendered_dir.rglob("*.svg")):
        png_path = svg_path.with_suffix(".png")
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            dpi=dpi,
        )
        print(f"  wrote {png_path}")
        rendered += 1
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
