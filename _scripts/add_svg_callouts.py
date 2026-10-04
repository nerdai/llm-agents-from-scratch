"""Overlay annotation callouts onto a rendered PlantUML class diagram.

A callout is a small dashed box ("class method", "enum", ...) joined to
the thing it labels by a dashed elbow leader line, in the style of
Manning figure annotations. PlantUML's own `note` cannot do this well:
a note on a class member draws a wedge connector, a note on a link
cannot point at the link's diamond, and the layout engine centres a
note beside a tall class instead of beside the member it is about.
This draws the callouts directly as SVG on top of the finished render
instead, driven by a sidecar YAML file next to the `.puml` source, the
same way `add_svg_legend.py` draws legends.

Sidecar file naming: `<diagram>.callouts.yaml` next to `<diagram>.puml`,
e.g. `uml/ch03/llm_class.callouts.yaml` for `uml/ch03/llm_class.puml`.
Format:

    callouts:
      - text: class method
        anchor: {text: from_tool_call_result}
        side: bottom
        at: 0.15
        box: [-200, 450]
      - text: composition
        anchor: {link: composition}
        box: [0, 520]
      - text: enum
        anchor: {entity: ChatRole}
        side: bottom
        box: [260, 300]

`anchor` picks what the leader starts from:

- `{text: <substring>}`: the first `<text>` element containing the
  substring, e.g. a member's name. Its box is the text's extent. Add
  `nth: 2` (1-based) to pick a later match, e.g. a constructor
  parameter that repeats an attribute.
- `{entity: <name>}`: the class (or enum, ...) whose qualified name ends
  with `.<name>`, or equals it. Its box is the class's outline.
- `{link: <type>}`: the first link of that PlantUML type (`composition`,
  `aggregation`, ...). The anchor point is the centre of its diamond.

For text and entity anchors, `side` (`top`, `bottom`, `left`, `right`;
default `bottom`) picks the edge the leader leaves from, and `at` (0 to
1, default 0.5) how far along that edge. `box` is the callout box's
centre, as an `[dx, dy]` offset from the anchor point in SVG user units
(the raw PlantUML canvas, 500 dpi). The leader runs along the side's
axis first, then turns once into the nearer edge of the box, or runs
straight in when the box is lined up with the anchor.

`text` may span lines: write it as a YAML list of lines.

`also` lists extra anchors (each with its own `anchor`, `side`
and `at`) that get their own leader into the same box, e.g. an
attribute and the constructor parameter that sets it. The box is still
placed relative to the first anchor.

Boxes that extend past the canvas grow it, so a callout is never
clipped: right/down by enlarging the viewBox, left/up by also shifting
the diagram and callouts over inside a nested group.
`set_svg_print_size.py`, which runs later, scales the grown canvas like
any other.
"""

import re
from pathlib import Path
from typing import Any, NamedTuple

import fire
import yaml

FONT_FAMILY = "Times"  # book-clean's ANNO_FONT, used for its notes
FONT_SIZE = 83.3333  # book-clean's 16pt NoteFontSize @ 500dpi
CHAR_WIDTH = 0.5 * FONT_SIZE  # Times averages ~0.5em, so boxes fit
LINE_HEIGHT = 1.15 * FONT_SIZE
PADDING_X = 45.0
PADDING_Y = 30.0
STROKE = "stroke:#333333;stroke-width:5.2083;stroke-dasharray:26,18;"
FILL = "#FFFFCC"
CANVAS_MARGIN = 40.0
STRAIGHT_TOLERANCE = 1.0  # within this, the box counts as lined up

STARTUML_NAME = re.compile(r"@startuml\s+(\S+)")
VIEWBOX = re.compile(r'viewBox="0 0 ([\d.]+) ([\d.]+)"')
SVG_CLOSE = re.compile(r"</g></svg>")
CONTENT_GROUP = re.compile(r"(<\?plantuml[^>]*\?><defs/>)<g>")
TEXT = re.compile(r"<text\b([^>]*)>([^<]*)</text>")
ATTR = re.compile(r'([\w-]+)="([^"]*)"')
GROUP = re.compile(r'<g class="(entity|link)"([^>]*)>(.*?)</g>', re.S)
RECT = re.compile(r"<rect\b([^>]*)/?>")
POLYGON = re.compile(r'<polygon\b[^>]*points="([^"]*)"')
ENTITIES = {
    "&lt;": "<",
    "&gt;": ">",
    "&amp;": "&",
    "&#171;": "«",
    "&#187;": "»",
}


class Box(NamedTuple):
    """An axis-aligned box in SVG user units."""

    left: float
    top: float
    right: float
    bottom: float


def _attrs(raw: str) -> dict[str, str]:
    return dict(ATTR.findall(raw))


def _unescape(text: str) -> str:
    for entity, char in ENTITIES.items():
        text = text.replace(entity, char)
    return text


def _text_box(content: str, needle: str, nth: int = 1) -> Box:
    seen = 0
    for raw_attrs, raw_text in TEXT.findall(content):
        if needle not in _unescape(raw_text):
            continue
        seen += 1
        if seen < nth:
            continue
        a = _attrs(raw_attrs)
        x, y = float(a["x"]), float(a["y"])
        size = float(a.get("font-size", FONT_SIZE))
        width = float(a.get("textLength", len(raw_text) * size * 0.5))
        return Box(x, y - 0.8 * size, x + width, y + 0.2 * size)
    raise ValueError(f"no match {nth} for <text> containing {needle!r}")


def _entity_box(content: str, name: str) -> Box:
    for kind, raw_attrs, body in GROUP.findall(content):
        if kind != "entity":
            continue
        qualified = _attrs(raw_attrs).get("data-qualified-name", "")
        if qualified != name and not qualified.endswith(f".{name}"):
            continue
        rect = RECT.search(body)
        if rect is None:
            break
        a = _attrs(rect.group(1))
        x, y = float(a["x"]), float(a["y"])
        return Box(x, y, x + float(a["width"]), y + float(a["height"]))
    raise ValueError(f"no entity named {name!r}")


def _link_point(content: str, link_type: str) -> tuple[float, float]:
    for kind, raw_attrs, body in GROUP.findall(content):
        if kind != "link":
            continue
        if _attrs(raw_attrs).get("data-link-type") != link_type:
            continue
        polygon = POLYGON.search(body)
        if polygon is None:
            break
        nums = [float(n) for n in re.split(r"[ ,]+", polygon.group(1).strip())]
        xs, ys = nums[0::2], nums[1::2]
        return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    raise ValueError(f"no {link_type!r} link with a diamond")


def _edge_point(box: Box, side: str, at: float) -> tuple[float, float]:
    if side == "bottom":
        return box.left + (box.right - box.left) * at, box.bottom
    if side == "top":
        return box.left + (box.right - box.left) * at, box.top
    if side == "left":
        return box.left, box.top + (box.bottom - box.top) * at
    if side == "right":
        return box.right, box.top + (box.bottom - box.top) * at
    raise ValueError(f"side must be top, bottom, left or right: {side!r}")


def _anchor(
    content: str,
    spec: dict[str, Any],
) -> tuple[tuple[float, float], bool]:
    """Return the anchor point and whether the leader leaves vertically."""
    anchor = spec["anchor"]
    if "link" in anchor:
        return _link_point(content, anchor["link"]), True
    if "text" in anchor:
        box = _text_box(content, anchor["text"], int(anchor.get("nth", 1)))
    elif "entity" in anchor:
        box = _entity_box(content, anchor["entity"])
    else:
        raise ValueError(f"anchor needs text, entity or link: {anchor!r}")
    side = spec.get("side", "bottom")
    return _edge_point(box, side, float(spec.get("at", 0.5))), side in {
        "top",
        "bottom",
    }


def _leader(
    start: tuple[float, float],
    box: Box,
    vertical_first: bool,
) -> list[tuple[float, float]]:
    """Elbow path from start into the nearer edge of the box."""
    sx, sy = start
    cx, cy = (box.left + box.right) / 2, (box.top + box.bottom) / 2
    if vertical_first:
        if (
            box.left - STRAIGHT_TOLERANCE
            <= sx
            <= box.right + STRAIGHT_TOLERANCE
        ):
            return [start, (sx, box.top if cy > sy else box.bottom)]
        edge = box.left if cx > sx else box.right
        return [start, (sx, cy), (edge, cy)]
    if box.top - STRAIGHT_TOLERANCE <= sy <= box.bottom + STRAIGHT_TOLERANCE:
        return [start, (box.left if cx > sx else box.right, sy)]
    edge = box.top if cy > sy else box.bottom
    return [start, (cx, sy), (cx, edge)]


def _label_lines(spec: dict[str, Any]) -> list[str]:
    text = spec["text"]
    lines = text if isinstance(text, list) else [text]
    return [str(line) for line in lines]


def _callout_svg(content: str, spec: dict[str, Any]) -> tuple[str, Box]:
    start, vertical_first = _anchor(content, spec)
    dx, dy = spec["box"]
    lines = _label_lines(spec)
    width = max(len(line) for line in lines) * CHAR_WIDTH + PADDING_X * 2
    height = LINE_HEIGHT * len(lines) + PADDING_Y * 2
    cx, cy = start[0] + float(dx), start[1] + float(dy)
    box = Box(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
    parts = []
    leaders = [(start, vertical_first)]
    leaders += [_anchor(content, extra) for extra in spec.get("also", [])]
    for point, vertical in leaders:
        path = _leader(point, box, vertical)
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in path)
        parts.append(
            f'<polyline fill="none" points="{points}" style="{STROKE}"/>',
        )
    parts.append(
        f'<rect fill="{FILL}" x="{box.left:.2f}" y="{box.top:.2f}" '
        f'width="{width:.2f}" height="{height:.2f}" style="{STROKE}"/>',
    )
    for i, line in enumerate(lines):
        baseline = box.top + PADDING_Y + LINE_HEIGHT * i + 0.8 * FONT_SIZE
        parts.append(
            f'<text fill="#000000" font-family="\'{FONT_FAMILY}\'" '
            f'font-size="{FONT_SIZE}" text-anchor="middle" '
            f'x="{cx:.2f}" y="{baseline:.2f}">{line}</text>',
        )
    return "".join(parts), box


def _apply_one(svg_path: Path, callouts_path: Path) -> bool:
    """Draw the callouts described by callouts_path onto svg_path.

    Returns:
        bool: True if the SVG was modified.
    """
    content = svg_path.read_text()
    viewbox_match = VIEWBOX.search(content)
    if not viewbox_match:
        return False
    width, height = float(viewbox_match.group(1)), float(viewbox_match.group(2))
    spec = yaml.safe_load(callouts_path.read_text())
    parts, boxes = [], []
    for callout in spec["callouts"]:
        svg, box = _callout_svg(content, callout)
        parts.append(svg)
        boxes.append(box)
    # Boxes left of or above the canvas: shift the drawing (diagram and
    # callouts together) right/down by the overhang, inside a nested group
    # so the outer `<defs/><g>` anchor survives for later pipeline steps.
    shift_x = max(0.0, *(CANVAS_MARGIN - b.left for b in boxes))
    shift_y = max(0.0, *(CANVAS_MARGIN - b.top for b in boxes))
    grown_w = max([width, *(b.right + CANVAS_MARGIN for b in boxes)]) + shift_x
    grown_h = (
        max([height, *(b.bottom + CANVAS_MARGIN for b in boxes)]) + shift_y
    )
    overlay = f'<g data-callouts="1">{"".join(parts)}</g>'
    tail = overlay + "</g></svg>"
    if shift_x or shift_y:
        content, moved = CONTENT_GROUP.subn(
            rf'\1<g><g transform="translate({shift_x:.4f},{shift_y:.4f})">',
            content,
            count=1,
        )
        if not moved:
            return False
        tail = overlay + "</g></g></svg>"
    if (grown_w, grown_h) != (width, height):
        content = VIEWBOX.sub(
            f'viewBox="0 0 {grown_w:.4f} {grown_h:.4f}"',
            content,
            count=1,
        )
    fixed, count = SVG_CLOSE.subn(tail, content, count=1)
    if not count:
        return False
    svg_path.write_text(fixed)
    return True


def main(rendered_dir: Path | str, uml_dir: Path | str = "uml") -> None:
    """Overlay callouts onto every rendered SVG with a sidecar spec.

    Args:
        rendered_dir (Path | str): Directory containing rendered SVGs
            (mirrors the chapter structure under uml_dir).
        uml_dir (Path | str): Directory containing `.puml` sources and
            their `*.callouts.yaml` sidecar files.

    Examples:
        >>> uv run python _scripts/add_svg_callouts.py \
        ...     --rendered_dir uml/rendered
    """
    rendered_dir = Path(rendered_dir)
    uml_dir = Path(uml_dir)
    applied = 0
    for callouts_path in sorted(uml_dir.rglob("*.callouts.yaml")):
        puml_path = callouts_path.with_name(
            callouts_path.name.replace(".callouts.yaml", ".puml"),
        )
        name_match = STARTUML_NAME.search(puml_path.read_text())
        if not name_match:
            continue
        svg_path = (
            rendered_dir
            / callouts_path.relative_to(uml_dir).parent
            / f"{name_match.group(1)}.svg"
        )
        if svg_path.exists() and _apply_one(svg_path, callouts_path):
            applied += 1
            print(f"  callouts: {svg_path}")
    print(f"Added callouts to {applied} diagram(s).")


if __name__ == "__main__":
    fire.Fire(main)
