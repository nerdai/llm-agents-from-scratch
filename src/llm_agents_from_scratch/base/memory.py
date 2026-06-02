"""Base memory and memory store classes."""

from abc import ABC, abstractmethod
from typing import Any

from ..data_structures import Episode, Task


class BaseMemory(ABC):
    """Base class for episodic memory.

    Subclasses are responsible for three things:

    1. **Retrieval strategy** — given a task and a store, decide which store
       primitive(s) to invoke and how to compose or score the results.
    2. **Prompt formatting** — convert the retrieved episodes into the string
       that gets injected into the agent's system prompt.
    3. **Write-time transformation** — optionally transform an episode before
       recording it (e.g. reflection, summarisation, fact extraction).
    """

    @abstractmethod
    async def recall(self, task: Task) -> str:
        """Retrieve relevant past episodes for a task.

        Called at the start of each task. Returns a formatted string ready to
        inject into the agent's system prompt under the recalled_episodes block.

        Args:
            task (Task): The incoming task used to select relevant episodes.

        Returns:
            str: Formatted episode context for the system prompt.
        """

    @abstractmethod
    async def record(self, episode: Episode) -> None:
        """Persist a completed episode.

        Called at the end of each task. Implementations may apply write-time
        transformations (e.g. reflection, summarisation) before storing.

        Args:
            episode (Episode): The completed task, trajectory, and result.
        """

    @abstractmethod
    async def summary(self) -> str:
        """Return a human-readable summary of the memory store.

        Useful for inspection and debugging — describes what is currently
        held in memory (e.g. episode count, date range, topics).

        Returns:
            str: A short summary of the memory contents.
        """


class BaseMemoryStore(ABC):
    """Base class for episodic memory stores.

    Subclasses are responsible for two things:

    1. **Durable persistence** — write episodes to the substrate, including
       any preprocessing required by that substrate (embedding, indexing,
       tokenization).
    2. **Retrieval primitives** — expose `read_recent` and `search` so that
       `BaseMemory` implementations can compose and score results without
       knowing the underlying storage details.

    Attributes:
        max_results (int): Default number of results returned by
            ``search`` and ``read_recent`` when no explicit count is
            supplied by the caller.
    """

    def __init__(self, max_results: int = 5) -> None:
        """Initialise shared store state.

        Args:
            max_results (int): Default maximum number of episodes
                returned by retrieval operations. Defaults to 5.
        """
        self.max_results = max_results

    @abstractmethod
    async def write(
        self,
        episode: Episode,
        key: str | None = None,
    ) -> None:
        """Persist an episode to the store.

        Args:
            episode (Episode): The completed episode to store.
            key (str | None): Pre-formatted text for substrates that
                embed episodes (e.g. vector stores). When provided, the
                store uses this text for embedding instead of formatting
                the episode itself. Substrates that do not embed (e.g.
                ``JSONMemoryStore``) ignore this argument. Defaults to
                ``None``.
        """

    @abstractmethod
    async def read_recent(self, n: int) -> list[Episode]:
        """Return the N most recently recorded episodes.

        Args:
            n (int): Maximum number of episodes to return.

        Returns:
            list[Episode]: Episodes ordered from most recent to oldest.
        """

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of episodes in the store.

        Returns:
            int: Episode count.
        """

    @abstractmethod
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

    @abstractmethod
    async def summary(self) -> str:
        """Return a human-readable summary of the store contents.

        Describes the substrate, episode count, and the oldest and newest
        episodes. Intended for inspection and debugging.

        Returns:
            str: Multi-line summary of the store.
        """
