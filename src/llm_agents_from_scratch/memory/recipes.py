"""Standard memory recipes for common episodic memory patterns."""

from pathlib import Path

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.data_structures.memory import Episode
from llm_agents_from_scratch.memory.memory import Memory, MetadataFn
from llm_agents_from_scratch.memory_stores.json import JSONMemoryStore
from llm_agents_from_scratch.memory_stores.qdrant.store import QdrantMemoryStore

_REFLECTION_TEMPLATE = """\
You just completed the following task.

Task: {instruction}
Result: {result}

In one sentence, what is the key lesson or actionable insight from this \
experience that would help with similar tasks in the future?\
"""


def _reflect_fn(llm: BaseLLM) -> MetadataFn:
    async def _reflect(episode: Episode) -> str:
        prompt = _REFLECTION_TEMPLATE.format(
            instruction=episode.task.instruction,
            result=episode.result.content,
        )
        result = await llm.complete(prompt)
        return result.response  # type: ignore[no-any-return]

    return _reflect


def recency_memory(
    path: Path,
    n: int = 5,
) -> Memory:
    """Return a recency-based episodic memory backed by a JSONL file.

    Recalls the ``n`` most recently recorded episodes. No embedding or
    similarity search is performed.

    Args:
        path (Path): Directory in which the backing JSONL file is stored.
        n (int): Maximum number of recent episodes to recall. Defaults
            to 5.

    Returns:
        Memory: Configured memory instance.
    """
    return Memory(store=JSONMemoryStore(dir=path, max_results=n))


def similarity_memory(
    collection: str = "episodes",
    k: int = 5,
) -> Memory:
    """Return a similarity-based episodic memory backed by Qdrant.

    Recalls the ``k`` most semantically similar past episodes using
    cosine similarity over FastEmbed vectors.

    Args:
        collection (str): Name of the Qdrant collection. Defaults to
            ``"episodes"``.
        k (int): Maximum number of similar episodes to recall. Defaults
            to 5.

    Returns:
        Memory: Configured memory instance.
    """
    return Memory(
        store=QdrantMemoryStore(collection_name=collection, max_results=k),
    )


def reflective_memory(
    path: Path,
    llm: BaseLLM,
    n: int = 5,
) -> Memory:
    """Return a reflective episodic memory backed by a JSONL file.

    Implements the Reflexion pattern: at record time an LLM distils a
    one-sentence lesson from the episode and stores it under
    ``episode.metadata["reflection"]``, which is then surfaced on
    recall via ``Episode.format()``.

    Args:
        path (Path): Directory in which the backing JSONL file is stored.
        llm (BaseLLM): LLM used to generate the reflection.
        n (int): Maximum number of recent episodes to recall. Defaults
            to 5.

    Returns:
        Memory: Configured memory instance.
    """
    return Memory(
        store=JSONMemoryStore(dir=path, max_results=n),
        metadata_fns={"reflection": _reflect_fn(llm)},
    )
