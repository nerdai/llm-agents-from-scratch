"""Single concrete Memory class."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from llm_agents_from_scratch.base.memory_store import BaseMemoryStore
from llm_agents_from_scratch.data_structures import Task
from llm_agents_from_scratch.data_structures.memory import Episode

MetadataFn = Callable[[Episode], str | Awaitable[str]]


class Memory:
    """Episodic memory: owns a store, a key function, and metadata pipeline.

    At record time, ``metadata_fns`` are run concurrently and their results
    written into ``episode.metadata`` before the episode is persisted.
    ``key_fn`` extracts the text used by the store for embedding/indexing.

    At recall time, ``store.search`` is called with the task instruction and
    episodes are formatted for prompt injection.

    Attributes:
        store (BaseMemoryStore): The underlying store that handles
            persistence and retrieval.
        key_fn (Callable[[Episode], str]): Callable that extracts the
            search key from an episode. Used as the embedding text by
            vector stores. Defaults to ``lambda ep: ep.task.instruction``.
        metadata_fns (dict[str, MetadataFn]): Mapping of metadata key
            to callable. Each callable receives the episode and returns
            a string written into ``episode.metadata[key]`` at record
            time. Callables may be sync or async.
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        key_fn: Callable[[Episode], str] | None = None,
        metadata_fns: dict[str, MetadataFn] | None = None,
    ) -> None:
        """Initialise a Memory instance.

        Args:
            store (BaseMemoryStore): The memory store to read from and
                write to.
            key_fn (Callable[[Episode], str] | None): Extracts the text
                used for embedding/indexing from the episode. Defaults
                to ``lambda ep: ep.task.instruction``.
            metadata_fns (dict[str, MetadataFn] | None): Optional
                mapping of metadata key to sync or async callable.
                Each callable receives the episode and its return value
                is stored under ``episode.metadata[key]``. All
                callables run concurrently at record time. Defaults to
                ``{}``.
        """
        self.store = store
        self.key_fn = (
            key_fn if key_fn is not None else lambda ep: ep.task.instruction
        )
        self.metadata_fns = metadata_fns or {}

    async def recall(self, task: Task) -> str:
        """Retrieve relevant past episodes for a task.

        Calls ``store.search(task.instruction)`` and formats the result
        as a newline-separated XML string for injection into the system
        prompt.

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
        """Persist a completed episode.

        Runs all ``metadata_fns`` concurrently, writes their results
        into ``episode.metadata``, then persists the episode using
        ``key_fn(episode)`` as the embedding key.

        Args:
            episode (Episode): The completed episode to enrich and
                store.
        """
        if self.metadata_fns:

            async def _call(fn: MetadataFn) -> str:
                if inspect.iscoroutinefunction(fn):
                    return await fn(episode)  # type: ignore[no-any-return]
                return fn(episode)  # type: ignore[return-value]

            values = await asyncio.gather(
                *[_call(fn) for fn in self.metadata_fns.values()],
            )
            for key, value in zip(
                self.metadata_fns.keys(),
                values,
                strict=True,
            ):
                episode.metadata[key] = value

        await self.store.write(episode, self.key_fn(episode))

    async def summary(self) -> str:
        """Return a human-readable summary of this memory instance.

        Combines the backing store's summary with the names of the
        configured ``key_fn`` and ``metadata_fns`` so the full
        configuration is visible at a glance.

        Returns:
            str: Multi-line summary of the store contents and memory
                configuration.
        """
        store_summary: str = await self.store.summary()  # type: ignore[no-any-return]
        key_fn_name = getattr(self.key_fn, "__name__", repr(self.key_fn))
        lines = [store_summary]
        lines.append(f"  key_fn: {key_fn_name}")
        if self.metadata_fns:
            keys = ", ".join(self.metadata_fns.keys())
            lines.append(f"  metadata_fns: {keys}")
        else:
            lines.append("  metadata_fns: none")
        return "\n".join(lines)

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
