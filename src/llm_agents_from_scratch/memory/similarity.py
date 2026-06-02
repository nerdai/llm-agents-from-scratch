"""Similarity-based episodic memory."""

from typing import Any

from llm_agents_from_scratch.base.memory import BaseMemory, BaseMemoryStore
from llm_agents_from_scratch.data_structures import Task
from llm_agents_from_scratch.data_structures.memory import Episode


class SimilarityMemory(BaseMemory):
    """Similarity-based episodic memory.

    Recalls the K most semantically similar past episodes using the
    store's vector search. At write time, episodes are passed to the
    store as-is; embedding is handled by the store.

    Attributes:
        store (BaseMemoryStore): The underlying vector memory store.
        k (int): Number of most similar episodes to recall.
        search_kwargs (dict[str, Any]): Extra keyword arguments forwarded
            to ``store.search()`` on every recall (e.g. score thresholds
            or payload filters for Qdrant).
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        k: int = 3,
        search_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a SimilarityMemory.

        Args:
            store (BaseMemoryStore): A vector-backed memory store that
                implements ``search()``.
            k (int): Number of most similar episodes to include in
                recall. Defaults to 3.
            search_kwargs (dict[str, Any] | None): Optional keyword
                arguments forwarded to ``store.search()`` on each
                recall. Defaults to an empty dict.
        """
        self.store = store
        self.k = k
        self.search_kwargs = search_kwargs or {}

    async def recall(self, task: Task) -> str:
        """Retrieve the K most semantically similar past episodes.

        Uses the task instruction as the search query.

        Args:
            task (Task): The incoming task whose instruction is used as
                the similarity query.

        Returns:
            str: Newline-separated episode strings, or empty string if
                no similar episodes are found.
        """
        episodes = await self.store.search(
            query=task.instruction,
            **self.search_kwargs,
        )
        if not episodes:
            return ""
        return "\n".join(str(ep) for ep in episodes)

    async def record(self, episode: Episode) -> None:
        """Persist a completed episode to the store.

        Args:
            episode (Episode): The completed episode to store.
        """
        await self.store.write(episode)

    async def summary(self) -> str:
        """Return a human-readable summary of the memory and its store.

        Describes the retrieval strategy (top-K similarity), then
        delegates to the store for substrate-level facts (count,
        oldest, newest).

        Returns:
            str: Multi-line summary of the memory and its store.
        """
        store_summary = await self.store.summary()
        return f"SimilarityMemory (top-{self.k} similarity):\n{store_summary}"
