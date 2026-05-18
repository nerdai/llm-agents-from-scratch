"""Recency-based episodic memory."""

from llm_agents_from_scratch.base.memory import BaseMemory, BaseMemoryStore
from llm_agents_from_scratch.data_structures import Task
from llm_agents_from_scratch.data_structures.memory import Episode


class RecencyMemory(BaseMemory):
    """Recency-based episodic memory.

    Recalls the last N episodes from the store and formats them for prompt
    injection. The simplest possible memory implementation: append on write,
    last-N on recall.

    Attributes:
        store (BaseMemoryStore): The underlying memory store.
        n (int): Number of most recent episodes to recall.
    """

    def __init__(self, store: BaseMemoryStore, n: int = 3) -> None:
        """Initialize a RecencyMemory.

        Args:
            store (BaseMemoryStore): The memory store to read from and write
                to.
            n (int): Number of most recent episodes to include in recall.
                Defaults to 3.
        """
        self.store = store
        self.n = n

    async def recall(self, task: Task) -> str:
        """Retrieve the N most recent episodes as a formatted string.

        Args:
            task (Task): The incoming task (unused — recency recall is
                unconditional).

        Returns:
            str: Newline-separated episode strings, or empty string if the
                store is empty.
        """
        episodes = await self.store.read_recent(self.n)
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

        Describes the recall strategy (last N), then delegates to the
        store for substrate-level facts (count, oldest, newest).

        Returns:
            str: Multi-line summary of the memory and its store.
        """
        store_summary = await self.store.summary()
        return f"RecencyMemory (recall last {self.n}):\n{store_summary}"
