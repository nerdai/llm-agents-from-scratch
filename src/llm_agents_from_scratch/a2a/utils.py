"""A2A data-conversion utilities."""

from collections.abc import Iterable

from a2a.types import Part, Task


def parts_text(parts: Iterable[Part]) -> str:
    """Join the text of every text Part, skipping non-text parts.

    Args:
        parts (Iterable[Part]): Parts to extract text from.

    Returns:
        str: The joined text, newline-separated.
    """
    return "\n".join(p.text for p in parts if p.text)


def task_content(task: Task) -> str:
    """Concatenate text from every artifact on a completed Task.

    Args:
        task (Task): The task to extract artifact text from.

    Returns:
        str: The joined text across all artifacts, newline-separated.
    """
    parts = [p for artifact in task.artifacts for p in artifact.parts]
    return parts_text(parts)
