"""JSONL-file-backed episodic memory store."""

from pathlib import Path
from typing import Any

from llm_agents_from_scratch.base.memory import BaseMemoryStore
from llm_agents_from_scratch.data_structures.memory import Episode


class JSONMemoryStore(BaseMemoryStore):
    """Episodic memory store backed by a JSONL file on disk.

    Each episode is stored as one JSON object per line (JSONL format).
    New episodes are appended to the file — no full rewrite on each write.
    The full file is read into memory on construction and on ``read_recent``.

    Attributes:
        path (Path): Full path to the backing JSONL file (``dir / filename``).
    """

    def __init__(
        self,
        dir: Path,
        filename: str = "episodes.jsonl",
    ) -> None:
        """Initialize a JSONMemoryStore.

        Loads any existing episodes from ``dir / filename`` on construction.
        The file is created on the first ``write`` call if it does not yet
        exist.

        Args:
            dir (Path): Directory in which the backing JSONL file is stored.
                The caller decides the location; no default is imposed by the
                library.
            filename (str): Name of the JSONL file within ``dir``. Defaults
                to ``"episodes.jsonl"``.
        """
        self.path = dir / filename
        self._episodes: list[Episode] = self._load()

    def _load(self) -> list[Episode]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [
                Episode.model_validate_json(line) for line in f if line.strip()
            ]

    async def write(self, episode: Episode) -> None:
        """Persist an episode to the store.

        Appends one JSON line to the backing file. Does not rewrite existing
        content.

        Args:
            episode (Episode): The completed episode to store.
        """
        self._episodes.append(episode)
        with open(self.path, "a") as f:
            f.write(episode.model_dump_json() + "\n")

    async def read_recent(self, n: int) -> list[Episode]:
        """Return the N most recently recorded episodes.

        Reads all episodes from the in-memory list. A production
        implementation would tail the file to avoid reading it in full.

        Args:
            n (int): Maximum number of episodes to return.

        Returns:
            list[Episode]: Episodes ordered from most recent to oldest.
        """
        return sorted(
            self._episodes,
            key=lambda e: e.completed_at,
            reverse=True,
        )[:n]

    async def count(self) -> int:
        """Return the total number of episodes in the store.

        Returns:
            int: Episode count.
        """
        return len(self._episodes)

    async def search(
        self,
        query: str,
        k: int,
        **kwargs: Any,
    ) -> list[Episode]:
        """Not implemented — similarity search is deferred to vector store.

        Args:
            query (str): The search query.
            k (int): Maximum number of episodes to return.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Always. Use a vector-backed store for
                similarity search.
        """
        raise NotImplementedError(
            "JSONMemoryStore does not support similarity search. "
            "Use a vector-backed store instead.",
        )
