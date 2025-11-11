from unittest.mock import MagicMock, patch

from llm_agents_from_scratch.data_structures import Task, TaskStep


@patch("llm_agents_from_scratch.data_structures.agent.uuid")
def test_string_representation_task(mock_uuid: MagicMock) -> None:
    """Test conversion of tool call result to an ChatMessage."""
    mock_uuid.uuid4.return_value = "111"
    task = Task(
        instruction="a fake instruction",
    )

    assert task.id_ == "111"
    assert str(task) == "a fake instruction"


@patch("llm_agents_from_scratch.data_structures.agent.uuid")
def test_string_representation_task_step(mock_uuid: MagicMock) -> None:
    """Test conversion of tool call result to an ChatMessage."""
    mock_uuid.uuid4.return_value = "111"
    task_step = TaskStep(
        instruction="a fake instruction",
        task_id="000",
    )

    assert str(task_step) == "a fake instruction"
    assert task_step.task_id == "000"
    assert task_step.id_ == "111"
