"""Data structures for episodic memory."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .agent import Task, TaskResult


class RecallMode(str, Enum):
    """Retrieval strategy used by ``BaseMemoryStore.search()``."""

    RECENT = "recent"
    SEARCH = "search"


class EpisodeFormatMode(str, Enum):
    """Serialisation format used by ``Episode.format()``."""

    XML = "xml"
    CONCAT = "concat"


EpisodeAttr = Literal[
    "instruction",
    "result",
    "error",
    "metadata",
    "completed_at",
    "rollout",
]


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
        include: list[EpisodeAttr] | None = None,
    ) -> str:
        """Serialise the episode for prompt injection or embedding.

        Args:
            mode (EpisodeFormatMode): ``EpisodeFormatMode.XML`` produces a
                prompt-ready XML block; ``EpisodeFormatMode.CONCAT`` produces
                a newline-joined string for embedding. Defaults to
                ``EpisodeFormatMode.XML``.
            include (list[EpisodeAttr] | None): Attributes to include.
                Defaults to ``["instruction", "result",
                "metadata", "completed_at"]``.

        Returns:
            str: Serialised episode string.
        """
        # rollout is excluded by default — it is long and adds noise
        attrs = include or [
            "instruction",
            "result",
            "error",
            "metadata",
            "completed_at",
        ]
        if mode == EpisodeFormatMode.CONCAT:
            return self._format_concat(attrs)
        return self._format_xml(attrs)

    def _format_concat(self, fields: list[EpisodeAttr]) -> str:
        parts: list[str] = []
        for f in fields:
            if f == "instruction":
                parts.append(f"instruction: {self.task.instruction}")
            elif f == "result" and self.result:
                parts.append(f"result: {self.result.content}")
            elif f == "error" and self.error:
                parts.append(f"error: {self.error}")
            elif f == "metadata" and self.metadata:
                parts.extend(f"{k}: {v}" for k, v in self.metadata.items())
            elif f == "completed_at":
                ts = self.completed_at.strftime("%Y-%m-%d %H:%M:%S")
                parts.append(f"completed_at: {ts}")
            elif f == "rollout":
                parts.append(f"rollout: {self.rollout}")
        return "\n".join(parts)

    def _format_xml(self, fields: list[EpisodeAttr]) -> str:
        lines = ["  <episode>"]
        for f in fields:
            if f == "instruction":
                lines.append(
                    f"    <task>{self.task.instruction}</task>",
                )
            elif f == "result" and self.result:
                lines.append(
                    f"    <result>{self.result.content}\n    </result>",
                )
            elif f == "error" and self.error:
                lines.append(
                    f"    <error>{self.error}\n    </error>",
                )
            elif f == "metadata" and self.metadata:
                for key, val in self.metadata.items():
                    lines.append(f"    <{key}>{val}</{key}>")
            elif f == "completed_at":
                ts = self.completed_at.strftime("%Y-%m-%d %H:%M:%S")
                lines.append(f"    <completed_at>{ts}</completed_at>")
            elif f == "rollout":
                lines.append(f"    <rollout>{self.rollout}</rollout>")
        lines.append("  </episode>")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return a prompt-ready XML string representation of the episode."""
        return self.format(mode=EpisodeFormatMode.XML)
