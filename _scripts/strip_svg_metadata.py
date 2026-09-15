"""Strip PlantUML's processing instructions from rendered SVGs.

PlantUML writes two processing instructions into every SVG: a version
tag, and `<?plantuml-src ...?>` carrying the whole diagram source
compressed into roughly a kilobyte of text. Both sit inside the `<svg>`
element rather than before it.

Canva refuses to import an SVG that contains either of them, reporting
only that the file "is not compatible with Canva or has been
corrupted". The files are valid: they parse as XML, and a DOM-level
comparison against a working copy shows no difference at all. That is
the tell -- the offending content is something a DOM parser discards,
which is to say a processing instruction.

This also explains why the hand-edited figures never hit the problem.
Inkscape rebuilds a file from its parsed DOM on save, so it dropped the
instructions silently; uploading pipeline output directly is what
exposed it.

Both instructions have to go. PlantUML's own `-nometadata` flag removes
only the source one and the result is still refused.

Nothing depends on them here. `<?plantuml-src ...?>` exists so a diagram
can be recovered from its own SVG, which this repo does not need: the
`.puml` files are the source and `uml/rendered/` is a gitignored build
artifact.
"""

import re
from pathlib import Path

import fire

#: Matches a processing instruction, but never an XML declaration.
PROCESSING_INSTRUCTION = re.compile(r"<\?(?!xml[\s?])[^?]*\?>")


def _strip_one(path: Path) -> int:
    """Remove processing instructions from a single SVG.

    Args:
        path (Path): The SVG to rewrite in place.

    Returns:
        int: How many instructions were removed.
    """
    content = path.read_text()
    stripped, count = PROCESSING_INSTRUCTION.subn("", content)
    if count:
        path.write_text(stripped)
    return count


def main(rendered_dir: Path | str) -> None:
    """Strip processing instructions from every SVG under rendered_dir.

    Args:
        rendered_dir (Path | str): Directory to search for `.svg` files
            (recursively).

    Examples:
        >>> uv run python _scripts/strip_svg_metadata.py \
        ...     --rendered_dir uml/rendered
    """
    rendered_dir = Path(rendered_dir)
    removed = 0
    touched = 0
    for svg_path in sorted(rendered_dir.rglob("*.svg")):
        count = _strip_one(svg_path)
        if count:
            touched += 1
            removed += count
    print(
        f"Stripped {removed} processing instruction(s) "
        f"from {touched} SVG file(s).",
    )


if __name__ == "__main__":
    fire.Fire(main)
