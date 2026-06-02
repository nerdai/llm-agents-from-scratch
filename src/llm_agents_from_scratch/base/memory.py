"""Base memory class."""

from abc import ABC, abstractmethod

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
