import re
from unittest.mock import patch

import pytest

from llm_agents_from_scratch.errors import MissingExtraError
from llm_agents_from_scratch.notebook_utils._check_installation import (
    check_notebooks_utils_installed,
)


def test_check_raises_error() -> None:
    """Check raises error from utils."""

    modules = {"pandas": None}

    with patch.dict("sys.modules", modules):
        msg = (
            "The `notebook-utils` extra is required for this function. "
            "Install with `pip install "
            "llm-agents-from-scratch[notebook-utils]`."
        )
        with pytest.raises(
            MissingExtraError,
            match=re.escape(msg),
        ):
            check_notebooks_utils_installed()
