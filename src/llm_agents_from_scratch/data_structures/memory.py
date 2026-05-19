"""Data structures for episodic memory."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .agent import Task, TaskResult

EpisodeAttr = Literal[
    "instruction",
    "result",
    "additional_data",
    "completed_at",
    "rollout",
]

_XML_DEFAULTS: list[EpisodeAttr] = [
    "instruction",
    "result",
    "additional_data",
    "completed_at",
]
_CONCAT_DEFAULTS: list[EpisodeAttr] = [
    "instruction",
    "result",
    "additional_data",
]


class Episode(BaseModel):
    """A completed task to be stored in memory.

    Attributes:
        task (Task): The task that was executed.
        rollout (str): The full agent trajectory for the task.
        result (TaskResult): The final result of the task.
        additional_data (dict[str, str] | None): Optional key-value
            annotations added by memory strategies at write time (e.g.
            ``{"reflection": "..."}`` from ReflectiveMemory).
        completed_at (datetime): Timestamp when the episode was recorded.
    """

    task: Task
    rollout: str
    result: TaskResult
    additional_data: dict[str, str] | None = None
    completed_at: datetime = Field(default_factory=datetime.now)

    def format(
        self,
        mode: Literal["xml", "concat"] = "xml",
        include: list[EpisodeAttr] | None = None,
    ) -> str:
        """Serialise the episode for prompt injection or embedding.

        Args:
            mode (Literal["xml", "concat"]): ``"xml"`` produces a
                prompt-ready XML block; ``"concat"`` produces a
                newline-joined string for embedding. Defaults to
                ``"xml"``.
            include (list[EpisodeAttr] | None): Attributes to include.
                Defaults to ``["instruction", "result",
                "additional_data", "completed_at"]`` for ``"xml"`` and
                ``["instruction", "result", "additional_data"]`` for
                ``"concat"``.

        Returns:
            str: Serialised episode string.
        """
        if mode == "concat":
            return self._format_concat(include or _CONCAT_DEFAULTS)
        return self._format_xml(include or _XML_DEFAULTS)

    def _format_concat(self, fields: list[EpisodeAttr]) -> str:
        parts: list[str] = []
        for f in fields:
            if f == "instruction":
                parts.append(self.task.instruction)
            elif f == "result":
                parts.append(self.result.content)
            elif f == "additional_data" and self.additional_data:
                parts.extend(self.additional_data.values())
            elif f == "completed_at":
                parts.append(
                    self.completed_at.strftime("%Y-%m-%d %H:%M:%S"),
                )
            elif f == "rollout":
                parts.append(self.rollout)
        return "\n".join(parts)

    def _format_xml(self, fields: list[EpisodeAttr]) -> str:
        lines = ["  <episode>"]
        for f in fields:
            if f == "instruction":
                lines.append(
                    f"    <task>{self.task.instruction}</task>",
                )
            elif f == "result":
                lines.append(
                    f"    <result>{self.result.content}\n    </result>",
                )
            elif f == "additional_data" and self.additional_data:
                for key, val in self.additional_data.items():
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
        return self.format(mode="xml")
