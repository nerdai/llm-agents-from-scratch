"""Skills constants."""

from pathlib import Path

SKILLS_DIRS = [
    ".from_scratch/skills",  # llm-agents-from-scratch skills dir
    ".agents/skills",  # cross-client skill sharing
]

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
