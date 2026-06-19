import warnings

import pytest

from llm_agents_from_scratch.base.memory_store import BaseMemoryStore
from llm_agents_from_scratch.data_structures import Task, TaskResult
from llm_agents_from_scratch.data_structures.memory import Episode, RecallMode
from llm_agents_from_scratch.errors import MaxResultsExceededWarning


def test_base_memory_store_abstract_methods() -> None:
    """Tests abstract methods of BaseMemoryStore."""
    abstract_methods = BaseMemoryStore.__abstractmethods__

    assert "write" in abstract_methods
    assert "count" in abstract_methods
    assert "_read_recent" in abstract_methods
    assert "_search" in abstract_methods
    assert "summary" in abstract_methods


def _make_episode() -> Episode:
    task = Task(instruction="test task")
    return Episode(
        task=task,
        rollout="",
        result=TaskResult(task_id=task.id_, content="test result"),
    )


class StubStore(BaseMemoryStore):
    """Minimal concrete store for testing BaseMemoryStore.recall() dispatch."""

    def __init__(
        self,
        episodes: list[Episode],
        recall_mode: RecallMode = RecallMode.SEARCH,
        max_results: int = 5,
    ) -> None:
        super().__init__(max_results=max_results, recall_mode=recall_mode)
        self._episodes = episodes

    async def write(self, episode: Episode) -> None:
        pass

    async def _read_recent(self, n: int) -> list[Episode]:
        return self._episodes[:n]

    async def _search(self, query: str, **kwargs: object) -> list[Episode]:
        return self._episodes

    async def count(self) -> int:
        return len(self._episodes)

    async def delete(self, id_: str) -> None:
        pass

    async def update(self, episode: Episode) -> None:
        pass

    async def summary(self) -> str:
        return ""


@pytest.mark.asyncio
async def test_recall_dispatches_to_search_when_recall_mode_search() -> None:
    episodes = [_make_episode(), _make_episode()]
    store = StubStore(episodes, recall_mode=RecallMode.SEARCH)

    result = await store.recall("query")

    assert result == episodes


@pytest.mark.asyncio
async def test_recall_dispatches_to_read_recent_when_recall_mode_recent() -> (
    None
):
    episodes = [_make_episode(), _make_episode(), _make_episode()]
    store = StubStore(episodes, recall_mode=RecallMode.RECENT, max_results=2)

    result = await store.recall("ignored")

    assert result == episodes[:2]


@pytest.mark.asyncio
async def test_recall_warns_when_results_exceed_max_results() -> None:
    episodes = [_make_episode(), _make_episode(), _make_episode()]
    store = StubStore(episodes, recall_mode=RecallMode.SEARCH, max_results=2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await store.recall("query")

    assert len(caught) == 1
    assert issubclass(caught[0].category, MaxResultsExceededWarning)
    assert "3" in str(caught[0].message)
    assert "max_results=2" in str(caught[0].message)


@pytest.mark.asyncio
async def test_recall_no_warning_when_results_within_max_results() -> None:
    episodes = [_make_episode(), _make_episode()]
    store = StubStore(episodes, recall_mode=RecallMode.SEARCH, max_results=5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await store.recall("query")

    assert not any(
        issubclass(w.category, MaxResultsExceededWarning) for w in caught
    )
