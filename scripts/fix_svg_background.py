"""Fix transparent backgrounds in PlantUML-generated SVGs.

PlantUML (since ~v1.2023+) emits `background:#FFFFFF` as a CSS style hint
on the root `<svg>` element instead of drawing an actual background
rectangle. Many SVG viewers don't honor CSS `background` on the SVG root
(only real drawn content), so the diagram renders with a transparent
background instead of white -- invisible text/lines against a dark or
checkered viewer background.

This inserts a literal `<rect>` covering the canvas as the first drawn
element, which every SVG renderer honors.
"""

import re
from pathlib import Path

import fire

BACKGROUND_RECT = '<rect width="100%" height="100%" fill="#FFFFFF"/>'
SVG_OPEN_TAG = re.compile(r"<svg[^>]*>")


def _fix_one(path: Path) -> bool:
    """Insert a background rect into a single SVG file if not already present.

    Returns:
        bool: True if the file was modified, False if already fixed or no
            `<svg>` tag was found.
    """
    content = path.read_text()
    if BACKGROUND_RECT in content:
        return False
    match = SVG_OPEN_TAG.search(content)
    if not match:
        return False
    fixed = content[: match.end()] + BACKGROUND_RECT + content[match.end() :]
    path.write_text(fixed)
    return True


def main(rendered_dir: Path | str) -> None:
    """Insert an opaque background rect into every SVG under rendered_dir.

    Args:
        rendered_dir (Path | str): Directory to search for `.svg` files
            (recursively).

    Examples:
        >>> uv run python scripts/fix_svg_background.py \
        ...     --rendered_dir uml/rendered
    """
    rendered_dir = Path(rendered_dir)
    fixed = 0
    for svg_path in rendered_dir.rglob("*.svg"):
        if _fix_one(svg_path):
            fixed += 1
    print(f"Fixed background on {fixed} SVG file(s) under {rendered_dir}.")


if __name__ == "__main__":
    fire.Fire(main)
