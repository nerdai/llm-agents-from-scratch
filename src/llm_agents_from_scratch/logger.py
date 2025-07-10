"""LLM Agents From Scratch Library Logger."""

import logging
import sys

from colorama import Fore, Style, init
from typing_extensions import override

# Initialize colorama for cross-platform colored output
init(autoreset=True)

ROOT_LOGGER_NAME = "llm_agents_fs"
DEFAULT_LOG_LEVEL = logging.INFO


class ColoredFormatter(logging.Formatter):
    """Colored formatter for logging."""

    COLORS = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.MAGENTA,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    @override
    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        original_msg = record.getMessage()
        logger_name = record.name

        colored_levelname = (
            f"{self.COLORS.get(levelname, '')}{levelname}{Style.RESET_ALL}"
        )

        return f"{colored_levelname} ({logger_name}) :      {original_msg}"


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger for the library.

    Args:
        name: Optional module name (will be prefixed with library name)

    Returns:
        Logger instance with NullHandler by default
    """
    logger_name = f"{ROOT_LOGGER_NAME}.{name}" if name else ROOT_LOGGER_NAME
    logger = logging.getLogger(logger_name)

    # Add NullHandler only to root library logger if no handlers exist
    if logger_name == ROOT_LOGGER_NAME and not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger


def enable_console_logging(level: str | int = DEFAULT_LOG_LEVEL) -> None:
    """Enable colored console logging for the library.

    Args:
        level: Logging level (e.g., "INFO", "DEBUG", logging.INFO)
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    library_logger = logging.getLogger(ROOT_LOGGER_NAME)
    library_logger.setLevel(level)

    # Remove existing console handlers to avoid duplicates
    for handler in library_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            library_logger.removeHandler(handler)

    # Add new console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    library_logger.addHandler(console_handler)


def disable_console_logging() -> None:
    """Disable console logging for the library."""
    library_logger = logging.getLogger(ROOT_LOGGER_NAME)

    # Remove console handlers
    for handler in library_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            library_logger.removeHandler(handler)

    # Ensure NullHandler exists
    if not library_logger.handlers:
        library_logger.addHandler(logging.NullHandler())
