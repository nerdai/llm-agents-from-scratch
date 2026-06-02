"""Single concrete Memory class."""

from collections.abc import Awaitable, Callable

from llm_agents_from_scratch.base.memory import BaseMemoryStore
from llm_agents_from_scratch.data_structures import Task
from llm_agents_from_scratch.data_structures.memory import Episode

TransformFn = Callable[[Episode], Awaitable[Episode]]


class Memory:
    """Episodic memory: owns a store and a write-time transformation pipeline.

    Recall delegates directly to the store's ``search`` method, using the
    task instruction as the query and ``store.max_results`` as the limit.
    The transformation pipeline is applied in order at write time before
    the episode is persisted (e.g. reflection, summarisation).

    Construct instances via ``MemoryBuilder`` or the convenience factories
    that will be introduced alongside it.

    Attributes:
        store (BaseMemoryStore): The underlying store that handles
            persistence and retrieval.
        transformations (list[TransformFn]): Write-time callables applied
            in order before ``store.write()``. Each receives an ``Episode``
            and returns the (possibly mutated) ``Episode``.
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        transformations: list[TransformFn] | None = None,
    ) -> None:
        """Initialise a Memory instance.

        Args:
            store (BaseMemoryStore): The memory store to read from and
                write to.
            transformations (list[TransformFn] | None): Optional
                write-time callables. Each receives an ``Episode`` and
                returns the transformed ``Episode``. Applied in order
                before ``store.write()``. Defaults to ``[]``.
        """
        self.store = store
        self.transformations = (
            transformations if transformations is not None else []
        )

    async def recall(self, task: Task) -> str:
        """Retrieve relevant past episodes for a task.

        Calls ``store.search(task.instruction, store.max_results)`` and
        formats the result as a newline-separated XML string for injection
        into the system prompt.

        Args:
            task (Task): The incoming task used to search for relevant
                episodes.

        Returns:
            str: Formatted episode context, or an empty string if no
                episodes are available.
        """
        episodes = await self.store.search(task.instruction)
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

        Requires the backing store to implement ``delete()``. Raises
        ``NotImplementedError`` until ``BaseMemoryStore.delete()`` is
        added (see issue #585).

        Args:
            id_ (str): The ``id_`` of the episode to delete.
        """
        raise NotImplementedError(
            f"{type(self.store).__name__} does not support delete().",
        )

    async def update(self, episode: Episode) -> None:
        """Replace an existing episode in the store.

        Requires the backing store to implement ``update()``. Raises
        ``NotImplementedError`` until ``BaseMemoryStore.update()`` is
        added (see issue #585).

        Args:
            episode (Episode): The episode to update. Matched by the
                episode identifier once ``Episode.id_`` is available
                (see issue #584).
        """
        raise NotImplementedError(
            f"{type(self.store).__name__} does not support update().",
        )
