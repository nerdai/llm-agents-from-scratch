"""Overlay a numbered legend box onto a rendered PlantUML SVG.

PlantUML's own layout primitives (`legend`, `note`) can only be pinned
to the corners of the whole canvas or to a specific point in the
message flow -- neither lets a legend sit at an arbitrary spot (e.g.
the top-right corner of the lifelines) without disturbing the layout.
This draws the legend directly as SVG on top of the finished render
instead, driven by a small sidecar YAML file next to the `.puml`
source.

Sidecar file naming: `<diagram>.legend.yaml` next to `<diagram>.puml`,
e.g. `uml/ch08/approval_gate_sequence.legend.yaml` for
`uml/ch08/approval_gate_sequence.puml`. Format:

    top: 620          # y position, in SVG user units
    entries:
      - label: "1"
        text: ask the human for approval

Busy diagrams may have no whitespace big enough to overlay a legend
into -- a seven-entry box needs ~780 user units of clear vertical run,
which a dense sequence diagram simply does not have at any width. Set
`placement: below` to append the legend under the diagram instead,
growing the canvas to make room. `top` is ignored in that mode.
Placement defaults to `overlay` so existing sidecars are unaffected.

`full_width: true` (only with `placement: below`) stretches the box
across the whole canvas instead of right-aligning a content-sized one,
so the entries read as a footnote strip under the figure. If the
entries need more room than the diagram is wide, the canvas is widened
and the diagram re-centred above the strip. Defaults to false.

`columns: N` lays the entries out in N columns, filled row-wise, which
trades height for width -- seven entries in three columns is three rows
rather than seven. Useful under `placement: below`, where the diagram's
full width is free and every extra row costs figure height. Defaults to
1, again leaving existing sidecars alone.
"""

import re
from pathlib import Path
from typing import Any

import fire
import yaml

FONT_FAMILY = "Arial"
FONT_SIZE = 72.9167  # matches book-clean's 14pt body text @ 500dpi
LINE_HEIGHT = 100.0
PADDING = 40.0
CHAR_WIDTH = 42.0  # heuristic; text is force-fit via textLength anyway
LABEL_COLUMN_CHARS = 3
MARGIN_RIGHT = 60.0

STARTUML_NAME = re.compile(r"@startuml\s+(\S+)")
VIEWBOX = re.compile(r'viewBox="0 0 ([\d.]+) ([\d.]+)"')
SVG_CLOSE = re.compile(r"</g></svg>")
CONTENT_GROUP = re.compile(r"(<\?plantuml[^>]*\?><defs/>)<g>")


def _line_length(label: str, text: str) -> float:
    return (LABEL_COLUMN_CHARS + len(text) + 1) * CHAR_WIDTH


def _column_widths(
    entries: list[dict[str, Any]],
    columns: int,
) -> list[float]:
    """Widest line in each column, filling row-wise."""
    return [
        max(_line_length(e["label"], e["text"]) for e in entries[col::columns])
        for col in range(columns)
    ]


def _legend_rows(entries: list[dict[str, Any]], columns: int) -> int:
    return -(-len(entries) // columns)


def _legend_height(entries: list[dict[str, Any]], columns: int = 1) -> float:
    return LINE_HEIGHT * _legend_rows(entries, columns) + PADDING * 2


def _build_legend_svg(
    entries: list[dict[str, Any]],
    canvas_width: float,
    top: float,
    columns: int = 1,
    full_width: bool = False,
) -> str:
    col_widths = _column_widths(entries, columns)
    box_height = _legend_height(entries, columns)
    if full_width:
        x = MARGIN_RIGHT
        box_width = canvas_width - MARGIN_RIGHT * 2
    else:
        box_width = sum(col_widths) + PADDING * 2
        # Right-aligned, but never off the left edge: a box wider than
        # the canvas would otherwise be silently clipped rather than
        # reported.
        x = max(MARGIN_RIGHT, canvas_width - box_width - MARGIN_RIGHT)
    y = top

    parts = [
        f'<rect fill="#FFFFFF" x="{x:.2f}" y="{y:.2f}" '
        f'width="{box_width:.2f}" height="{box_height:.2f}" '
        f'rx="31.25" ry="31.25" '
        f'style="stroke:#222222;stroke-width:2.6042;"/>',
    ]
    for i, entry in enumerate(entries):
        row, col = divmod(i, columns)
        label_x = x + PADDING + sum(col_widths[:col])
        text_x = label_x + LABEL_COLUMN_CHARS * CHAR_WIDTH
        line_y = y + PADDING + LINE_HEIGHT * row + FONT_SIZE
        label, text = str(entry["label"]), str(entry["text"])
        parts.append(
            f'<text fill="#000000" font-family="\'{FONT_FAMILY}\'" '
            f'font-weight="700" font-size="{FONT_SIZE}" '
            f'x="{label_x:.2f}" y="{line_y:.2f}">{label}</text>',
        )
        parts.append(
            f'<text fill="#000000" font-family="\'{FONT_FAMILY}\'" '
            f'font-size="{FONT_SIZE}" x="{text_x:.2f}" '
            f'y="{line_y:.2f}">{text}</text>',
        )
    return "".join(parts)


def _apply_one(svg_path: Path, legend_path: Path) -> bool:
    """Draw the legend described by legend_path onto svg_path.

    Returns:
        bool: True if the SVG was modified.
    """
    content = svg_path.read_text()
    viewbox_match = VIEWBOX.search(content)
    if not viewbox_match:
        return False
    canvas_width = float(viewbox_match.group(1))
    canvas_height = float(viewbox_match.group(2))

    spec = yaml.safe_load(legend_path.read_text())
    entries = spec["entries"]
    columns = int(spec.get("columns", 1))

    full_width = bool(spec.get("full_width", False))
    close_inner = False

    if spec.get("placement", "overlay") == "below":
        if full_width:
            needed = (
                sum(_column_widths(entries, columns))
                + PADDING * 2
                + MARGIN_RIGHT * 2
            )
            if needed > canvas_width:
                # Widen the canvas for the strip and re-centre the
                # diagram over it. The translate goes in a *nested*
                # group so the outer `<defs/><g>` anchor survives for
                # frame_svg_width.py, and so the strip appended after
                # it is not shifted along with the diagram.
                dx = (needed - canvas_width) / 2
                content, moved = CONTENT_GROUP.subn(
                    rf'\1<g><g transform="translate({dx:.4f},0)">',
                    content,
                    count=1,
                )
                if not moved:
                    return False
                canvas_width = needed
                close_inner = True
        top = canvas_height + PADDING
        # set_svg_print_size recomputes width/height from the viewBox, so
        # growing the viewBox alone is enough to reserve the space.
        grown = canvas_height + _legend_height(entries, columns) + PADDING * 2
        content = VIEWBOX.sub(
            f'viewBox="0 0 {canvas_width:.4f} {grown:.4f}"',
            content,
            count=1,
        )
    else:
        top = float(spec.get("top", 60))

    legend_svg = _build_legend_svg(
        entries,
        canvas_width,
        top=top,
        columns=columns,
        full_width=full_width,
    )
    tail = ("</g>" if close_inner else "") + legend_svg + "</g></svg>"
    fixed, count = SVG_CLOSE.subn(tail, content, count=1)
    if not count:
        return False
    svg_path.write_text(fixed)
    return True


def main(rendered_dir: Path | str, uml_dir: Path | str = "uml") -> None:
    """Overlay legends onto every rendered SVG with a sidecar spec.

    Args:
        rendered_dir (Path | str): Directory containing rendered SVGs
            (mirrors the chapter structure under uml_dir).
        uml_dir (Path | str): Directory containing `.puml` sources and
            their `*.legend.yaml` sidecar files.

    Examples:
        >>> uv run python _scripts/add_svg_legend.py \
        ...     --rendered_dir uml/rendered
    """
    rendered_dir = Path(rendered_dir)
    uml_dir = Path(uml_dir)
    applied = 0
    for legend_path in uml_dir.rglob("*.legend.yaml"):
        puml_path = legend_path.with_name(
            legend_path.name.replace(".legend.yaml", ".puml"),
        )
        name_match = STARTUML_NAME.search(puml_path.read_text())
        if not name_match:
            continue
        svg_path = (
            rendered_dir
            / legend_path.relative_to(uml_dir).parent
            / f"{name_match.group(1)}.svg"
        )
        if svg_path.exists() and _apply_one(svg_path, legend_path):
            applied += 1
    print(f"Applied legend overlay to {applied} SVG file(s).")


if __name__ == "__main__":
    fire.Fire(main)
