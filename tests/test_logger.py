"""Unit tests for the logging module."""

import logging

from llm_agents_from_scratch.logger import (
    ColoredFormatter,
    current_subagent_name,
)


def make_record(msg: str = "a message") -> logging.LogRecord:
    return logging.LogRecord(
        name="llm_agents_fs.TaskHandler",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_colored_formatter_no_prefix_by_default() -> None:
    """Tests format() omits the subagent prefix when unset."""
    formatted = ColoredFormatter().format(make_record())

    assert not formatted.startswith("\x1b[36m")


def test_colored_formatter_includes_subagent_prefix() -> None:
    """Tests format() prepends the subagent name when the contextvar is set."""
    token = current_subagent_name.set("hailstone")
    try:
        formatted = ColoredFormatter().format(make_record())
    finally:
        current_subagent_name.reset(token)

    assert formatted.startswith("\x1b[36m[hailstone]")


def test_colored_formatter_prefix_reverts_after_reset() -> None:
    """Tests the prefix disappears once the contextvar is reset."""
    token = current_subagent_name.set("hailstone")
    current_subagent_name.reset(token)

    formatted = ColoredFormatter().format(make_record())

    assert "[hailstone]" not in formatted
