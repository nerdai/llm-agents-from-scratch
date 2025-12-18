"""Utils for llm_agents_from_scratch."""

from importlib.util import find_spec

from llm_agents_from_scratch.errors import MissingExtraError


def check_extra_was_installed(extra: str, packages: str | list[str]) -> None:
    """Checks if extra is installed.

    Raises:
        MissingExtraError: If any of packages is not available, indicating
            that the specified extra has not been installed.
    """
    packages = [packages] if isinstance(packages, str) else packages
    if not all(find_spec(p) for p in packages):
        msg = (
            f"The `{extra}` extra is required for this function. "
            "Install with `pip install "
            f"llm-agents-from-scratch[{extra}]`."
        )
        raise MissingExtraError(msg)
