"""Base memory store class."""

from abc import ABC, abstractmethod
from typing import Any

from ..data_structures import Episode
from ..data_structures.memory import RecallMode


class BaseMemoryStore(ABC):
    """Base class for episodic memory stores.

    Subclasses are responsible for two things:

    1. **Durable persistence** — write episodes to the substrate, including
       any preprocessing required by that substrate (embedding, indexing,
       tokenization).
    2. **Retrieval primitives** — expose `read_recent` and `search` so that
       ``Memory`` instances can compose and score results without knowing
       the underlying storage details.

    Attributes:
        max_results (int): Default number of results returned by
            ``search`` and ``read_recent`` when no explicit count is
            supplied by the caller.
        recall_mode (RecallMode): Controls how ``search()`` retrieves
            episodes. ``RecallMode.RECENT`` ignores the query and
            returns the most recent episodes via ``read_recent``;
            ``RecallMode.SEARCH`` performs a similarity lookup.
    """

    def __init__(
        self,
        max_results: int = 5,
        recall_mode: RecallMode = RecallMode.SEARCH,
    ) -> None:
        """Initialise shared store state.

        Args:
            max_results (int): Default maximum number of episodes
                returned by retrieval operations. Defaults to 5.
            recall_mode (RecallMode): Retrieval strategy used by
                ``search()``. Defaults to ``RecallMode.SEARCH``.
        """
        self.max_results = max_results
        self.recall_mode = recall_mode

    @abstractmethod
    async def write(self, episode: Episode) -> None:
        """Persist an episode to the store.

        Args:
            episode (Episode): The completed episode to store.
        """

    @abstractmethod
    async def _read_recent(self, n: int) -> list[Episode]:
        """Return the N most recently recorded episodes.

        Args:
            n (int): Maximum number of episodes to return.

        Returns:
            list[Episode]: Episodes ordered from most recent to oldest.
        """

    @abstractmethod
    async def _search(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[Episode]:
        """Return the most relevant episodes for a query.

        The number of results is controlled by ``self.max_results``.

        Args:
            query (str): The search query (e.g. the task instruction).
            **kwargs: Optional substrate-specific search parameters
                (e.g. filters, score thresholds).

        Returns:
            list[Episode]: Episodes ordered by relevance to the query.
        """

    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[Episode]:
        """Return the most relevant episodes for a query.

        The number of results is controlled by ``self.max_results``.

        Args:
            query (str): The search query (e.g. the task instruction).
            **kwargs: Optional substrate-specific search parameters
                (e.g. filters, score thresholds).

        Returns:
            list[Episode]: Episodes ordered by relevance to the query.
        """
        if self.recall_mode == RecallMode.RECENT:
            return await self._read_recent(self.max_results)
        return await self._search(query, **kwargs)

    @abstractmethod
    async def delete(self, id_: str) -> None:
        """Delete an episode by its unique identifier.

        Raises ``EpisodeNotFoundError`` if no episode with ``id_`` exists.

        Args:
            id_ (str): The ``Episode.id_`` of the episode to remove.
        """

    @abstractmethod
    async def update(self, episode: Episode) -> None:
        """Replace an existing episode with an updated version.

        Matches the stored episode by ``episode.id_`` and replaces it
        in-place. Raises ``EpisodeNotFoundError`` if
        no matching episode exists.

        Args:
            episode (Episode): The updated episode. Matched by ``id_``.
        """

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of episodes in the store.

        Returns:
            int: Episode count.
        """

    @abstractmethod
    async def summary(self) -> str:
        """Return a human-readable summary of the store contents.

        Describes the substrate, episode count, and the oldest and newest
        episodes. Intended for inspection and debugging.

        Returns:
            str: Multi-line summary of the store.
        """
