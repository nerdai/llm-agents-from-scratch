"""Check installation of notebooks extra."""

from importlib.util import find_spec

from llm_agents_from_scratch.errors import MissingExtraError


def check_notebooks_utils_installed() -> None:
    """Checks if notebook-utils extra is installed."""
    if not find_spec("pandas"):
        msg = (
            "The `notebook-utils` extra is required for this function. "
            "Install with `pip install "
            "llm-agents-from-scratch[notebook-utils]`."
        )
        raise MissingExtraError(msg)
