"""Check installation of notebooks extra."""

from importlib.util import find_spec

from llm_agents_from_scratch.errors import MissingExtraError


def check_notebooks_utils_installed() -> None:
    """Checks if notebook-utils extra is installed.

    Raises:
        MissingExtraError: If the pandas package is not available, indicating
            that the notebook-utils extra has not been installed.
    """
    if not find_spec("pandas") or not find_spec("IPython"):
        msg = (
            "The `notebook-utils` extra is required for this function. "
            "Install with `pip install "
            "llm-agents-from-scratch[notebook-utils]`."
        )
        raise MissingExtraError(msg)
