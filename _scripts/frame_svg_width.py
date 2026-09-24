"""Center every rendered diagram SVG in a fixed-width frame.

`set_svg_print_size.py` gives each diagram its own correct physical
size, but that size still varies diagram to diagram (a small class
diagram might land at 2.6in wide, a dense one at 5.5in) -- placed in
the manuscript, that reads as inconsistent column alignment even
though the *font size* is now uniform. This widens every diagram's
canvas to a fixed frame (default 5.5in) and centers the actual
diagram horizontally within it, so every image is exactly the same
width on the page. Height is left alone -- only width is normalized.

Diagrams narrower than the frame are padded (whitespace added, content
re-centered) -- font size is untouched. A diagram wider than the frame
(shouldn't normally happen if `set_svg_print_size.py`'s own max width
is <= the frame width, but handled defensively) is scaled down to fit
exactly, which does shrink its font size slightly.

Padding is done by widening the SVG's `viewBox` and translating the
existing content group by half the added width -- not by drawing a
second white rectangle over it, so it composes cleanly with
`fix_svg_background.py`'s full-canvas background rect (already
percentage-sized, so it automatically covers the widened canvas too).

A `full_width` legend from `add_svg_legend.py` (marked with a
`<g data-full-width-legend="1">` wrapper) is a deliberate exception to
"content re-centered, not resized": it promises to span the print
frame, and that script runs before this one, so it can only size
itself against the canvas as it existed at that point -- narrower than
the frame whenever this script is about to pad it further. Left alone,
the legend would end up centered at its old, pre-pad width instead of
actually spanning the frame it was widened for. So the marker group
gets a counter-translate (cancelling the diagram-wide shift just for
that subtree) and its background rect is widened to the new frame,
same margins as add_svg_legend.py's own layout. MARGIN_RIGHT here must
match that script's constant of the same name.
"""

import re
from pathlib import Path

import fire

DEFAULT_FRAME_WIDTH_IN = 5.5
MARGIN_RIGHT = 60.0

VIEWBOX = re.compile(r'viewBox="0 0 ([\d.]+) ([\d.]+)"')
WIDTH_ATTR = re.compile(r'width="([\d.]+)in"')
HEIGHT_ATTR = re.compile(r'height="([\d.]+)in"')
STYLE_SIZE = re.compile(r"width:[\d.]+in;height:[\d.]+in;")
CONTENT_GROUP = re.compile(r"(<\?plantuml[^>]*\?><defs/>)<g>")
FULL_WIDTH_LEGEND = re.compile(
    r'(<g data-full-width-legend="1">)(<rect fill="#FFFFFF" x="[\d.]+" '
    r'y="[\d.]+" width=")([\d.]+)(")',
)


def _frame_one(
    svg_path: Path,
    frame_width_in: float,
) -> dict[str, float | str] | None:
    content = svg_path.read_text()
    viewbox_match = VIEWBOX.search(content)
    width_match = WIDTH_ATTR.search(content)
    height_match = HEIGHT_ATTR.search(content)
    if not (viewbox_match and width_match and height_match):
        return None

    view_w, view_h = (
        float(viewbox_match.group(1)),
        float(viewbox_match.group(2)),
    )
    width_in = float(width_match.group(1))
    height_in = float(height_match.group(1))

    if width_in > frame_width_in:
        # Defensive: shouldn't happen if set_svg_print_size.py's own
        # max width is already <= frame_width_in, but shrink to fit
        # rather than silently overflow the frame.
        scale = frame_width_in / width_in
        new_width_in = frame_width_in
        new_height_in = height_in * scale
        content = WIDTH_ATTR.sub(
            f'width="{new_width_in:.4f}in"',
            content,
            count=1,
        )
        content = HEIGHT_ATTR.sub(
            f'height="{new_height_in:.4f}in"',
            content,
            count=1,
        )
        content = STYLE_SIZE.sub(
            f"width:{new_width_in:.4f}in;height:{new_height_in:.4f}in;",
            content,
            count=1,
        )
        svg_path.write_text(content)
        return {
            "name": svg_path.name,
            "width_in": new_width_in,
            "height_in": new_height_in,
            "padded_in": 0.0,
            "shrunk": True,
        }

    pad_in = frame_width_in - width_in
    new_view_w = view_w * (frame_width_in / width_in)
    dx = (new_view_w - view_w) / 2

    content = VIEWBOX.sub(
        f'viewBox="0 0 {new_view_w:.4f} {view_h:.4f}"',
        content,
        count=1,
    )
    content, n = CONTENT_GROUP.subn(
        rf'\1<g transform="translate({dx:.4f},0)">',
        content,
        count=1,
    )
    if not n:
        return None
    content, legend_widened = FULL_WIDTH_LEGEND.subn(
        rf'<g data-full-width-legend="1" '
        rf'transform="translate({-dx:.4f},0)">'
        rf"\g<2>{new_view_w - 2 * MARGIN_RIGHT:.2f}\g<4>",
        content,
        count=1,
    )
    content = WIDTH_ATTR.sub(
        f'width="{frame_width_in:.4f}in"',
        content,
        count=1,
    )
    content = STYLE_SIZE.sub(
        f"width:{frame_width_in:.4f}in;height:{height_in:.4f}in;",
        content,
        count=1,
    )
    svg_path.write_text(content)

    return {
        "name": svg_path.name,
        "width_in": frame_width_in,
        "height_in": height_in,
        "padded_in": pad_in,
        "shrunk": False,
        "legend_widened": bool(legend_widened),
    }


def main(
    rendered_dir: Path | str,
    frame_width_in: float = DEFAULT_FRAME_WIDTH_IN,
) -> None:
    """Center every SVG under rendered_dir in a fixed-width frame.

    Args:
        rendered_dir (Path | str): Directory containing rendered SVGs
            (must already have physical width/height set, e.g. by
            set_svg_print_size.py).
        frame_width_in (float): Fixed frame width every diagram gets
            centered in, in inches.

    Examples:
        >>> uv run python _scripts/frame_svg_width.py \
        ...     --rendered_dir uml/rendered
    """
    rendered_dir = Path(rendered_dir)
    results = []
    for svg_path in sorted(rendered_dir.rglob("*.svg")):
        result = _frame_one(svg_path, frame_width_in)
        if result:
            results.append(result)

    shrunk = [r for r in results if r["shrunk"]]
    print(
        f"Framed {len(results)} SVG file(s) to a "
        f"{frame_width_in}in wide frame.",
    )
    if shrunk:
        print(
            f"{len(shrunk)} diagram(s) were wider than the frame and had to "
            "be shrunk to fit (check max_width_in in set_svg_print_size.py):",
        )
        for r in shrunk:
            print(
                f"  {r['name']}: {r['width_in']:.2f}in x "
                f"{r['height_in']:.2f}in",
            )
    widened = [r for r in results if r.get("legend_widened")]
    if widened:
        print(
            f"{len(widened)} full-width legend(s) re-widened to match "
            "the padded frame:",
        )
        for r in widened:
            print(f"  {r['name']}")


if __name__ == "__main__":
    fire.Fire(main)
