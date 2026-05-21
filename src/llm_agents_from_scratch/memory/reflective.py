"""Reflective episodic memory (Reflexion pattern)."""

from llm_agents_from_scratch.base.llm import BaseLLM
from llm_agents_from_scratch.base.memory import BaseMemoryStore
from llm_agents_from_scratch.data_structures.memory import Episode
from llm_agents_from_scratch.memory.recency import RecencyMemory

REFLECTION_TEMPLATE = """\
You just completed the following task.

Task: {instruction}
Result: {result}

In one sentence, what is the key lesson or actionable insight from this \
experience that would help with similar tasks in the future?\
"""


class ReflectiveMemory(RecencyMemory):
    """Reflexion-style episodic memory with write-time reflection.

    Implements the Reflexion pattern (https://arxiv.org/abs/2303.11366).
    Extends RecencyMemory with an LLM call at write time that distils a
    short lesson from each completed episode. The lesson is stored on
    the episode under ``additional_data["reflection"]`` and surfaced
    automatically in recall via ``Episode.format()``.

    Attributes:
        store (BaseMemoryStore): The underlying memory store.
        llm (BaseLLM): The LLM used to generate reflections at write
            time.
        n (int): Number of most recent episodes to recall.
        reflection_template (str): Template used to build the
            reflection prompt.
    """

    def __init__(
        self,
        store: BaseMemoryStore,
        llm: BaseLLM,
        n: int = 3,
        reflection_template: str = REFLECTION_TEMPLATE,
    ) -> None:
        """Initialize a ReflectiveMemory.

        Args:
            store (BaseMemoryStore): The memory store to read from and
                write to.
            llm (BaseLLM): The LLM used to generate a reflection after
                each episode completes.
            n (int): Number of most recent episodes to include in
                recall. Defaults to 3.
            reflection_template (str): Template string for the
                reflection prompt. Must contain ``{instruction}`` and
                ``{result}`` placeholders. Defaults to
                ``REFLECTION_TEMPLATE``.
        """
        super().__init__(store=store, n=n)
        self.llm = llm
        self.reflection_template = reflection_template

    async def record(self, episode: Episode) -> None:
        """Generate a reflection and persist the enriched episode.

        Calls the LLM to distil a one-sentence lesson from the episode,
        attaches it under ``additional_data["reflection"]``, then
        delegates to the store.

        Args:
            episode (Episode): The completed episode to reflect on and
                store.
        """
        prompt = self.reflection_template.format(
            instruction=episode.task.instruction,
            result=episode.result.content,
        )
        result = await self.llm.complete(prompt)
        episode.additional_data = {"reflection": result.response}
        await self.store.write(episode)

    async def summary(self) -> str:
        """Return a human-readable summary of the memory and its store.

        Returns:
            str: Multi-line summary of the memory and its store.
        """
        store_summary = await self.store.summary()
        return f"ReflectiveMemory (recall last {self.n}):\n{store_summary}"
