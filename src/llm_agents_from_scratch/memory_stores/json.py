"""JSONL-file-backed episodic memory store."""

from pathlib import Path
from typing import Any

from llm_agents_from_scratch.base.memory_store import BaseMemoryStore
from llm_agents_from_scratch.data_structures.memory import Episode, RecallMode
from llm_agents_from_scratch.errors import EpisodeNotFoundError


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
        max_results: int = 5,
        recall_mode: RecallMode = RecallMode.RECENT,
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
            max_results (int): Default maximum number of episodes returned
                by retrieval operations. Defaults to 5.
            recall_mode (RecallMode): Retrieval strategy used by
                ``search()``. Defaults to ``RecallMode.RECENT`` since
                this store does not support similarity search.
        """
        super().__init__(max_results=max_results, recall_mode=recall_mode)
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

    async def summary(self) -> str:
        """Return a human-readable summary of the store contents.

        Includes the backing file path, total episode count, and the
        instruction and timestamp of the newest and oldest episodes.

        Returns:
            str: Multi-line summary of the store.
        """
        total = len(self._episodes)
        lines = [f"JSONMemoryStore: {total} episodes | path={self.path}"]
        if total > 0:
            by_time = sorted(
                self._episodes,
                key=lambda e: e.completed_at,
            )
            oldest = by_time[0]
            newest = by_time[-1]
            lines.append(
                f"  newest: {str(newest.completed_at)[:19]}"
                f" | {newest.task.instruction[:60]}",
            )
            lines.append(
                f"  oldest: {str(oldest.completed_at)[:19]}"
                f" | {oldest.task.instruction[:60]}",
            )
        return "\n".join(lines)

    def _rewrite(self) -> None:
        with open(self.path, "w") as f:
            for ep in self._episodes:
                f.write(ep.model_dump_json() + "\n")

    async def delete(self, id_: str) -> None:
        """Delete an episode by its unique identifier.

        Raises ``EpisodeNotFoundError`` if no episode with ``id_`` exists.
        Otherwise removes it from the in-memory list and rewrites the
        backing file.

        Args:
            id_ (str): The ``Episode.id_`` of the episode to remove.

        Raises:
            EpisodeNotFoundError: If no episode with ``id_`` exists.
        """
        idx = next(
            (i for i, ep in enumerate(self._episodes) if ep.id_ == id_),
            None,
        )
        if idx is None:
            raise EpisodeNotFoundError(
                f"Episode '{id_}' not found in JSONMemoryStore.",
            )
        self._episodes.pop(idx)
        self._rewrite()

    async def update(self, episode: Episode) -> None:
        """Replace an existing episode with an updated version.

        Matches by ``episode.id_``. Raises ``EpisodeNotFoundError`` if no
        matching episode exists. Otherwise replaces it in-place and rewrites
        the backing file.

        Args:
            episode (Episode): The updated episode. Matched by ``id_``.

        Raises:
            EpisodeNotFoundError: If no episode with ``episode.id_`` exists.
        """
        for i, ep in enumerate(self._episodes):
            if ep.id_ == episode.id_:
                self._episodes[i] = episode
                self._rewrite()
                return
        raise EpisodeNotFoundError(
            f"Episode '{episode.id_}' not found in JSONMemoryStore.",
        )

    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[Episode]:
        """Return episodes according to ``recall_mode``.

        When ``recall_mode="recent"`` (the default), the query is ignored
        and the most recent episodes are returned via ``read_recent``.
        ``recall_mode="search"`` raises ``NotImplementedError`` — this store
        does not support similarity search.

        Args:
            query (str): The search query. Ignored when
                ``recall_mode="recent"``.
            **kwargs: Ignored.

        Returns:
            list[Episode]: Episodes ordered by recency.

        Raises:
            NotImplementedError: When ``recall_mode="search"``. Use a
                vector-backed store for similarity search.
        """
        if self.recall_mode == RecallMode.RECENT:
            return await self.read_recent(self.max_results)
        raise NotImplementedError(
            "JSONMemoryStore does not support similarity search. "
            "Use a vector-backed store instead.",
        )
