"""Thicken the edges from a build plan's current node back to the root.

`uml/build_plans/chNN.puml` marks its chapter's topic with the
`current` style class, which PlantUML renders as a doubled node border.
The matching emphasis on the *edges* leading back to the root cannot be
expressed in PlantUML: `arrow { LineThickness }` applies to every
connector at once, and scoping it to a class (`.current arrow { ... }`)
is silently a no-op. This applies that emphasis afterwards, so the
`.puml` files stay the single source of truth.

Nothing is passed in. PlantUML has already identified the current
node(s) by giving their `<rect>` a stroke-width above the base, so this
walks up from each one, collecting the connectors on the way to the
root, and doubles their stroke-width to match the border.

Only files named `build_plan_*.svg` are considered, which is what
`uml/build_plans/chNN.puml` produces via its `@startmindmap
build_plan_chNN` id. That guard is load-bearing rather than tidiness:
the current node is identified by having a thicker border than its
siblings, and several sequence diagrams legitimately mix rect
stroke-widths too. Without the filter this rewrites their connectors
and silently flattens lines that were meant to differ.
"""

import re
from pathlib import Path

import fire

#: Coordinate slack when matching a connector endpoint to a node edge.
#: Endpoints match exactly in practice; this only absorbs float
#: formatting in the emitted SVG.
TOLERANCE = 1.5

#: Guards against a malformed diagram sending the walk into a cycle.
MAX_HOPS = 20

#: Only these files are touched. See the module docstring -- this is a
#: safety guard, not a convenience.
BUILD_PLAN_PREFIX = "build_plan_"

RECT = re.compile(r"<rect[^>]*/>")
PATH = re.compile(r'<path d="([^"]+)"[^>]*?stroke-width:([\d.]+)')
COORD = re.compile(r"(-?[\d.]+),(-?[\d.]+)")
STROKE_WIDTH = re.compile(r"stroke-width:([\d.]+)")


def _rects(content: str) -> list[dict]:
    """Return every node box with numeric geometry and stroke-width.

    The full-canvas background rect inserted by `fix_svg_background.py`
    uses percentage widths, so it is skipped here.

    Args:
        content (str): Full SVG text.

    Returns:
        list[dict]: One entry per node box, with `x`, `y`, `width`,
            `height`, `sw` and the `span` of its tag in `content`.
    """
    found = []
    for match in RECT.finditer(content):
        tag = match.group(0)
        try:
            box = {
                key: float(re.search(rf'\b{key}="([\d.]+)"', tag).group(1))
                for key in ("x", "y", "width", "height")
            }
        except AttributeError:
            continue
        stroke = STROKE_WIDTH.search(tag)
        if not stroke:
            continue
        box["sw"] = float(stroke.group(1))
        box["span"] = match.span()
        found.append(box)
    return found


def _paths(content: str) -> list[dict]:
    """Return every connector with its start point, end point and width.

    Args:
        content (str): Full SVG text.

    Returns:
        list[dict]: One entry per connector, with `start`, `end`, `sw`
            and the `span` of its tag in `content`.
    """
    found = []
    for match in PATH.finditer(content):
        coords = COORD.findall(match.group(1))
        if not coords:
            continue
        found.append(
            {
                "start": (float(coords[0][0]), float(coords[0][1])),
                "end": (float(coords[-1][0]), float(coords[-1][1])),
                "sw": float(match.group(2)),
                "span": match.span(),
            },
        )
    return found


def _near(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Return True if two points coincide within `TOLERANCE`."""
    return abs(a[0] - b[0]) < TOLERANCE and abs(a[1] - b[1]) < TOLERANCE


def _edge_midpoints(box: dict) -> list[tuple[float, float]]:
    """Return a box's left and right edge midpoints, where edges attach."""
    middle = box["y"] + box["height"] / 2
    return [(box["x"], middle), (box["x"] + box["width"], middle)]


def _parent_anchors(
    point: tuple[float, float],
    towards_root: int,
    boxes: list[dict],
    connectors: list[dict],
) -> list[tuple[float, float]]:
    """Return where the parent node's own incoming edge would terminate.

    A connector starts at its parent's edge. To keep walking upward we
    need the *opposite* side of that parent, which is where the parent's
    own incoming connector lands.

    Two node shapes have to be handled. A boxed node has a `<rect>`, so
    the opposite edge is the other side of that rect. A boundless node
    (written `**_ Part 2`, drawn as bare text) has no rect at all; it
    spans two x positions on a shared baseline, so the opposite side is
    the other connector endpoint at the same `y`.

    A boundless node is matched on its shared baseline, which on its own
    is not selective enough: an unrelated node can sit at exactly that
    `y` on the far side of the tree. `towards_root` disambiguates by
    requiring the parent side to lie in the direction the root is in.

    Args:
        point (tuple[float, float]): Where the child's connector starts.
        towards_root (int): +1 if the root lies to the right of `point`,
            -1 if to the left.
        boxes (list[dict]): Node boxes from `_rects`.
        connectors (list[dict]): Connectors from `_paths`.

    Returns:
        list[tuple[float, float]]: Candidate endpoints for the parent's
            own incoming connector. Empty once the root is reached.
    """
    for box in boxes:
        middle = box["y"] + box["height"] / 2
        for x_edge in (box["x"], box["x"] + box["width"]):
            if _near((x_edge, middle), point):
                opposite = (
                    box["x"] + box["width"] if x_edge == box["x"] else box["x"]
                )
                return [(opposite, middle)]
    return [
        connector["end"]
        for connector in connectors
        if abs(connector["end"][1] - point[1]) < TOLERANCE
        and (connector["end"][0] - point[0]) * towards_root > TOLERANCE
    ]


def _bold_one(path: Path) -> int:
    """Double the ancestor-path connectors in a single SVG.

    Args:
        path (Path): The SVG to rewrite in place.

    Returns:
        int: How many connectors were thickened. Zero means the diagram
            has no current node, and the file is left untouched.
    """
    content = path.read_text()
    boxes, connectors = _rects(content), _paths(content)
    if not boxes or not connectors:
        return 0

    base = min(box["sw"] for box in boxes)
    current = [box for box in boxes if box["sw"] > base]
    if not current:
        return 0

    # A chapter whose topic has children marks all of them current, so
    # walk up from each; the paths converge and `seen` dedupes them.
    to_bold: dict[tuple[int, int], dict] = {}
    for box in current:
        anchors = _edge_midpoints(box)
        for _ in range(MAX_HOPS):
            step = next(
                (
                    connector
                    for connector in connectors
                    if connector["span"] not in to_bold
                    and any(_near(connector["end"], a) for a in anchors)
                ),
                None,
            )
            if step is None:
                break
            to_bold[step["span"]] = step
            # The step travelled away from the root, so the root lies
            # back in the opposite direction.
            towards_root = 1 if step["end"][0] < step["start"][0] else -1
            anchors = _parent_anchors(
                step["start"],
                towards_root,
                boxes,
                connectors,
            )
            if not anchors:
                break

    # Every edge is rewritten, not just the ones being thickened.
    # PlantUML propagates a node's LineThickness to its *outgoing*
    # connectors, so a current node with children already has thick
    # downward edges -- at the node base doubled, not the edge base, so
    # they do not even match. Normalising all of them guarantees exactly
    # two widths in the output and undoes that artifact where an edge is
    # not on an ancestor path.
    edge_base = min(connector["sw"] for connector in connectors)
    for connector in sorted(connectors, key=lambda c: -c["span"][0]):
        start, end = connector["span"]
        target = edge_base * 2 if connector["span"] in to_bold else edge_base
        if connector["sw"] == target:
            continue
        retyped = content[start:end].replace(
            f"stroke-width:{connector['sw']:g}",
            f"stroke-width:{target:g}",
        )
        content = content[:start] + retyped + content[end:]
    path.write_text(content)
    return len(to_bold)


def main(rendered_dir: Path | str) -> None:
    """Thicken ancestor-path edges in every SVG under rendered_dir.

    Args:
        rendered_dir (Path | str): Directory to search recursively.
            Only `build_plan_*.svg` files within it are modified.

    Examples:
        >>> uv run python _scripts/bold_current_path.py \
        ...     --rendered_dir uml/rendered
    """
    rendered_dir = Path(rendered_dir)
    changed = 0
    for svg_path in sorted(rendered_dir.rglob(f"{BUILD_PLAN_PREFIX}*.svg")):
        edges = _bold_one(svg_path)
        if edges:
            changed += 1
            print(f"  {svg_path.name}: {edges} edge(s) thickened")
    print(f"Thickened the current path in {changed} build plan(s).")


if __name__ == "__main__":
    fire.Fire(main)
