"""Task Handler"""

import asyncio
from typing import Any


class TaskHandler(asyncio.Future):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    async def run_step(last_step: Any) -> Any:
        """Run next step of a given task.

        Example: perform tool call, generated LLM response, etc.

        Args:
            last_step (Any): The result of the previous step.

        Returns:
            Any: The result of the next sub step and sets result if Task the completion
            of the sub-step represents the completion of the Task.
        """
        pass
