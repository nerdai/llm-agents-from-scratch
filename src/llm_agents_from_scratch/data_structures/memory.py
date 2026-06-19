"""Data structures for episodic memory."""

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .agent import Task, TaskResult


class RecallMode(str, Enum):
    """Retrieval strategy used by ``BaseMemoryStore.recall()``."""

    RECENT = "recent"
    SEARCH = "search"


class EpisodeFormatMode(str, Enum):
    """Serialization format used by ``Episode.format()``."""

    XML = "xml"
    CONCAT = "concat"


class Episode(BaseModel):
    """A completed task to be stored in memory.

    Attributes:
        id_ (str): Unique identifier for the episode (UUID4).
        task (Task): The task that was executed.
        rollout (str): The full agent trajectory for the task.
        result (TaskResult | None): The final result of the task, or
            ``None`` if the task ended with an error.
        error (Exception | None): The exception raised if the task
            failed, or ``None`` on success.
        metadata (dict[str, str]): Key-value annotations written at
            record time (e.g. ``{"reflection": "..."}``). Surfaced
            automatically in ``format()`` output.
        completed_at (datetime): Timestamp when the episode was recorded.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id_: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: Task
    rollout: str
    result: TaskResult | None = Field(default=None)
    error: Exception | None = Field(default=None)
    metadata: dict[str, str] = Field(default_factory=dict)
    completed_at: datetime = Field(default_factory=datetime.now)

    def format(
        self,
        mode: EpisodeFormatMode = EpisodeFormatMode.XML,
        exclude: set[str] | None = None,
    ) -> str:
        """Serialize the episode for prompt injection or embedding.

        Args:
            mode (EpisodeFormatMode): ``EpisodeFormatMode.XML`` produces a
                prompt-ready XML block; ``EpisodeFormatMode.CONCAT`` produces
                a newline-joined string for embedding. Defaults to
                ``EpisodeFormatMode.XML``.
            exclude (set[str] | None): Field names to omit. Defaults to
                ``{"id_", "rollout"}`` — both are excluded by default as
                they add noise in most contexts.

        Returns:
            str: Serialized episode string.
        """
        excluded = exclude if exclude is not None else {"id_", "rollout"}
        if mode == EpisodeFormatMode.CONCAT:
            return self._format_concat(excluded)
        return self._format_xml(excluded)

    def _format_concat(self, exclude: set[str]) -> str:
        parts: list[str] = []
        for f in Episode.model_fields:
            if f in exclude:
                continue
            if val := getattr(self, f):
                if f == "metadata":
                    parts.extend(f"{k}: {v}" for k, v in val.items())
                elif f == "completed_at":
                    ts = val.strftime("%Y-%m-%d %H:%M:%S")
                    parts.append(f"completed_at: {ts}")
                else:
                    parts.append(f"{f}: {val}")
        return "\n".join(parts)

    def _format_xml(self, exclude: set[str]) -> str:
        lines = ["  <episode>"]
        for f in Episode.model_fields:
            if f in exclude:
                continue
            if val := getattr(self, f):
                if f == "metadata":
                    for key, v in val.items():
                        lines.append(f"    <{key}>{v}</{key}>")
                elif f == "completed_at":
                    ts = val.strftime("%Y-%m-%d %H:%M:%S")
                    lines.append(f"    <completed_at>{ts}</completed_at>")
                else:
                    lines.append(f"    <{f}>{val}</{f}>")
        lines.append("  </episode>")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return a prompt-ready XML string representation of the episode."""
        return self.format(mode=EpisodeFormatMode.XML)
