from unittest.mock import MagicMock, patch

from llm_agents_from_scratch.data_structures import Task, TaskStep
from llm_agents_from_scratch.data_structures.agent import ApprovalResult


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


def test_approval_result_str_approved() -> None:
    """Tests ApprovalResult.__str__ when approved."""
    assert str(ApprovalResult(approved=True)) == "approved"


def test_approval_result_str_rejected() -> None:
    """Tests ApprovalResult.__str__ when rejected includes feedback."""
    result = ApprovalResult(approved=False, feedback="fix the math")
    assert str(result) == "rejected: fix the math"
