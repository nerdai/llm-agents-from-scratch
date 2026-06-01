"""Single concrete Memory class."""

from collections.abc import Awaitable, Callable

from llm_agents_from_scratch.base.memory import BaseMemoryStore
from llm_agents_from_scratch.data_structures import Task
from llm_agents_from_scratch.data_structures.memory import Episode

RecallFn = Callable[[Task], Awaitable[list[Episode]]]
TransformFn = Callable[[Episode], Awaitable[Episode]]


class Memory:
    """Episodic memory: owns a store, a recall function, and a pipeline.

    The recall function determines which episodes surface for a new task
    (e.g. most-recent or most-similar). The transformation pipeline is applied
    in order at write time before the episode is persisted (e.g. reflection,
    summarisation).

    Construct instances via the convenience factories (``RecencyMemory``,
    ``SimilarityMemory``, ``ReflectiveMemory``) or directly via
    ``MemoryBuilder``.

    Attributes:
        store (BaseMemoryStore): The underlying store that handles
            persistence and retrieval.
        recall_fn (RecallFn): Callable that receives a ``Task`` and returns
            the list of episodes to inject into the system prompt.
        transformations (list[TransformFn]): Write-time callables applied
            in order before ``store.write()``. Each receives an ``Episode``
            and returns the (possibly mutated) ``Episode``.
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        recall_fn: RecallFn,
        transformations: list[TransformFn] | None = None,
    ) -> None:
        """Initialise a Memory instance.

        Args:
            store (BaseMemoryStore): The memory store to read from and
                write to.
            recall_fn (RecallFn): Callable ``(task) -> list[Episode]``
                that selects which episodes to surface for a given task.
            transformations (list[TransformFn] | None): Optional
                write-time callables. Each receives an ``Episode`` and
                returns the transformed ``Episode``. Applied in order
                before ``store.write()``. Defaults to ``[]``.
        """
        self.store = store
        self.recall_fn = recall_fn
        self.transformations = transformations or []

    async def recall(self, task: Task) -> str:
        """Retrieve relevant past episodes for a task.

        Calls ``recall_fn(task)`` and formats the result as a
        newline-separated XML string for injection into the system prompt.

        Args:
            task (Task): The incoming task used to select relevant
                episodes.

        Returns:
            str: Formatted episode context, or an empty string if no
                episodes are available.
        """
        episodes = await self.recall_fn(task)
        if not episodes:
            return ""
        return "\n".join(str(ep) for ep in episodes)

    async def record(self, episode: Episode) -> None:
        """Persist a completed episode, applying any transformations first.

        Runs ``episode`` through each callable in ``transformations`` in
        order, then writes the result to the store.

        Args:
            episode (Episode): The completed episode to transform and
                store.
        """
        for transform in self.transformations:
            episode = await transform(episode)
        await self.store.write(episode)

    async def delete(self, id_: str) -> None:
        """Delete an episode from the store by its ID.

        Args:
            id_ (str): The ``id_`` of the episode to delete.
        """
        await self.store.delete(id_)

    async def update(self, episode: Episode) -> None:
        """Replace an existing episode in the store.

        Args:
            episode (Episode): The episode to update. Matched by
                ``episode.id_``.
        """
        await self.store.update(episode)
