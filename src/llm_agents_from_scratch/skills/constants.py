"""Skills constants."""

from pathlib import Path

SKILLS_PATHS: dict[str, list[Path]] = {
    "project": [
        Path.cwd() / ".from_scratch/skills",
        Path.cwd() / ".agent/skills",
    ],
    "user": [
        Path.home() / ".from_scratch/skills",
        Path.home() / ".agent/skills",
    ],
}

MAX_NAME_LENGTH = 64
