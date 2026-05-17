"""JSON-file-backed episodic memory store."""

import json
from pathlib import Path

from llm_agents_from_scratch.base.memory import BaseMemoryStore
from llm_agents_from_scratch.data_structures.memory import Episode


class JSONMemoryStore(BaseMemoryStore):
    """Episodic memory store backed by a JSON file on disk.

    Attributes:
        path (Path): Path to the JSON file used for persistence.
    """

    def __init__(self, path: Path) -> None:
        """Initialize a JSONMemoryStore.

        Loads any existing episodes from ``path`` on construction. The file
        is created on the first ``write`` call if it does not yet exist.

        Args:
            path (Path): Caller-supplied path to the backing JSON file. No
                default location is imposed by the library — the caller
                decides where episodes are stored.
        """
        self.path = path
        self._episodes: list[Episode] = self._load()

    def _load(self) -> list[Episode]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [Episode(**ep) for ep in json.load(f)]

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(
                [ep.model_dump(mode="json") for ep in self._episodes],
                f,
            )

    async def write(self, episode: Episode) -> None:
        """Persist an episode to the store.

        Appends to the in-memory list and rewrites the full JSON file.

        Args:
            episode (Episode): The completed episode to store.
        """
        self._episodes.append(episode)
        self._save()

    async def read_recent(self, n: int) -> list[Episode]:
        """Return the N most recently recorded episodes.

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

    async def search(self, query: str, k: int) -> list[Episode]:
        """Not implemented — similarity search is deferred to vector store.

        Args:
            query (str): The search query.
            k (int): Maximum number of episodes to return.

        Raises:
            NotImplementedError: Always. Use a vector-backed store for
                similarity search.
        """
        raise NotImplementedError(
            "JSONMemoryStore does not support similarity search. "
            "Use a vector-backed store instead.",
        )
