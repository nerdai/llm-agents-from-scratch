"""Set each rendered SVG's physical print size for consistent typography.

All diagrams share the same PlantUML theme (uml/common/book-clean.puml),
so every diagram's body text is rendered at the same font size in SVG
user units regardless of how large or complex the diagram is. If every
SVG is then fit to the same bounding box (e.g. by hand in Canva), a
diagram with a bigger viewBox -- more participants, more messages --
ends up with visibly smaller text than a simple diagram, because "fit
to box" scales relative to each diagram's own size, not to a shared
target font size.

This script fixes that by setting each SVG's `width`/`height`
attributes directly, in inches, so that placing the image at "actual
size" (100%, no further resizing) renders body text at a fixed target
point size everywhere -- unless the diagram is too large to hit that
target within the book's max figure box (default 5.6in x 7in), in
which case it is scaled down just enough to fit, and reported as such.

Class diagrams get a lower, uniform target than everything else
(CLASS_TARGET_PT, default 5.0pt vs DEFAULT_TARGET_PT's 9pt): they vary
far more in natural size (a two-class diagram next to a ten-method
one) than sequence diagrams do, so holding them to the shared 9pt
target left most of them capped anyway (17 of 22 at last count) while
a handful of simple ones rendered noticeably larger side by side in
the manuscript -- inconsistent either way. A floor picked as a
compromise (e.g. 7pt) still leaves a straggling tail of outliers below
it, which doesn't actually deliver a consistent reader experience,
just a smaller inconsistency. 5.0pt is calibrated to the single
worst-case class diagram's own natural ceiling (~4.92pt, a diagram
nearly 2x the width budget) -- rounded up slightly so effectively
every class diagram (21 of 22, the last one only ~0.1pt short) renders
at the exact same size, true uniformity within the existing 5.6in x
7in box, no diagram redesign required. Revisit upward once the
worst-offending diagrams are split/simplified (tracked separately).
"""

import re
from pathlib import Path

import fire

# Must match uml/common/book-clean.puml's skinparam dpi / DefaultFontSize.
THEME_DPI = 500
THEME_BODY_FONT_PT = 14

DEFAULT_TARGET_PT = 9.0
CLASS_TARGET_PT = 5.0
# 5.5in, not the book's full 5.6in figure-box width -- matches
# frame_svg_width.py's default frame, so no diagram needs a second,
# separate shrink pass once it's centered in that fixed-width frame.
DEFAULT_MAX_WIDTH_IN = 5.5
DEFAULT_MAX_HEIGHT_IN = 7.0

VIEWBOX = re.compile(r'viewBox="0 0 ([\d.]+) ([\d.]+)"')
WIDTH_ATTR = re.compile(r'width="[\d.]+px"')
HEIGHT_ATTR = re.compile(r'height="[\d.]+px"')
STYLE_SIZE = re.compile(r"width:[\d.]+px;height:[\d.]+px;")
DIAGRAM_TYPE = re.compile(r'data-diagram-type="([A-Z]+)"')


def _inches_per_user_unit(target_pt: float) -> float:
    user_units_per_pt = THEME_BODY_FONT_PT * (THEME_DPI / 96)
    return target_pt / (user_units_per_pt * 72)


def _resize_one(
    svg_path: Path,
    default_target_pt: float,
    class_target_pt: float,
    max_width_in: float,
    max_height_in: float,
) -> dict[str, float | str] | None:
    content = svg_path.read_text()
    match = VIEWBOX.search(content)
    if not match:
        return None
    view_w, view_h = float(match.group(1)), float(match.group(2))

    type_match = DIAGRAM_TYPE.search(content)
    diagram_type = type_match.group(1) if type_match else None
    target_pt = (
        class_target_pt if diagram_type == "CLASS" else default_target_pt
    )

    in_per_unit = _inches_per_user_unit(target_pt)
    natural_w_in = view_w * in_per_unit
    natural_h_in = view_h * in_per_unit

    fit_scale = min(
        1.0,
        max_width_in / natural_w_in,
        max_height_in / natural_h_in,
    )
    final_w_in = natural_w_in * fit_scale
    final_h_in = natural_h_in * fit_scale
    effective_pt = target_pt * fit_scale

    content = WIDTH_ATTR.sub(f'width="{final_w_in:.4f}in"', content, count=1)
    content = HEIGHT_ATTR.sub(f'height="{final_h_in:.4f}in"', content, count=1)
    content = STYLE_SIZE.sub(
        f"width:{final_w_in:.4f}in;height:{final_h_in:.4f}in;",
        content,
        count=1,
    )
    svg_path.write_text(content)

    return {
        "name": svg_path.name,
        "diagram_type": diagram_type or "?",
        "target_pt": target_pt,
        "width_in": final_w_in,
        "height_in": final_h_in,
        "effective_pt": effective_pt,
        "capped": fit_scale < 1.0,
    }


def main(
    rendered_dir: Path | str,
    target_pt: float = DEFAULT_TARGET_PT,
    class_target_pt: float = CLASS_TARGET_PT,
    max_width_in: float = DEFAULT_MAX_WIDTH_IN,
    max_height_in: float = DEFAULT_MAX_HEIGHT_IN,
) -> None:
    """Set physical print sizes on every SVG under rendered_dir.

    Args:
        rendered_dir (Path | str): Directory containing rendered SVGs.
        target_pt (float): Target body-text point size for non-class
            diagrams (sequence, mindmap, etc.) at print size.
        class_target_pt (float): Target body-text point size for class
            diagrams specifically -- lower and uniform, see module
            docstring for why class diagrams get their own target.
        max_width_in (float): Max figure width, in inches.
        max_height_in (float): Max figure height, in inches.

    Examples:
        >>> uv run python _scripts/set_svg_print_size.py \
        ...     --rendered_dir uml/rendered
    """
    rendered_dir = Path(rendered_dir)
    results = []
    for svg_path in sorted(rendered_dir.rglob("*.svg")):
        result = _resize_one(
            svg_path,
            target_pt,
            class_target_pt,
            max_width_in,
            max_height_in,
        )
        if result:
            results.append(result)

    capped = [r for r in results if r["capped"]]
    print(
        f"Set print size on {len(results)} SVG file(s) "
        f"(target {target_pt}pt, class diagrams {class_target_pt}pt).",
    )
    if capped:
        print(
            f"{len(capped)} diagram(s) exceed the {max_width_in}in x "
            f"{max_height_in}in box and were scaled down below target:",
        )
        for r in capped:
            print(
                f"  {r['name']}: {r['width_in']:.2f}in x "
                f"{r['height_in']:.2f}in @ {r['effective_pt']:.1f}pt",
            )


if __name__ == "__main__":
    fire.Fire(main)
