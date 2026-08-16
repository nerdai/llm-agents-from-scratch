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


def _line_length(label: str, text: str) -> float:
    return (LABEL_COLUMN_CHARS + len(text) + 1) * CHAR_WIDTH


def _build_legend_svg(
    entries: list[dict[str, Any]],
    canvas_width: float,
    top: float,
) -> str:
    box_width = max(_line_length(e["label"], e["text"]) for e in entries)
    box_width += PADDING * 2
    box_height = LINE_HEIGHT * len(entries) + PADDING * 2
    x = canvas_width - box_width - MARGIN_RIGHT
    y = top

    parts = [
        f'<rect fill="#FFFFFF" x="{x:.2f}" y="{y:.2f}" '
        f'width="{box_width:.2f}" height="{box_height:.2f}" '
        f'rx="31.25" ry="31.25" '
        f'style="stroke:#222222;stroke-width:2.6042;"/>',
    ]
    label_x = x + PADDING
    text_x = label_x + LABEL_COLUMN_CHARS * CHAR_WIDTH
    for i, entry in enumerate(entries):
        line_y = y + PADDING + LINE_HEIGHT * i + FONT_SIZE
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
    """Overlay the legend described by legend_path onto svg_path.

    Returns:
        bool: True if the SVG was modified.
    """
    content = svg_path.read_text()
    viewbox_match = VIEWBOX.search(content)
    if not viewbox_match:
        return False
    canvas_width = float(viewbox_match.group(1))

    spec = yaml.safe_load(legend_path.read_text())
    legend_svg = _build_legend_svg(
        spec["entries"],
        canvas_width,
        top=float(spec.get("top", 60)),
    )
    fixed, count = SVG_CLOSE.subn(legend_svg + "</g></svg>", content, count=1)
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
