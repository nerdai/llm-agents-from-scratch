"""LLM Agents From Scratch Library Logger."""

import logging
import sys

from colorama import Fore, Style, init
from typing_extensions import override

# Initialize colorama for cross-platform colored output
init(autoreset=True)

ROOT_LOGGER_NAME = "llm_agents_fs"  # truncate from scratch for brevity in logs
LOG_LEVEL = logging.DEBUG


_logger_configured = False


class ColoredFormatter(logging.Formatter):
    """Colored formatter for logging."""

    COLORS = {
        "DEBUG": Fore.BLUE,
        "INFO": Fore.MAGENTA,  # Using magenta instead of Flower's green
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Implements format.

        Args:
            record (logging.LogRecord): The record to format.

        Returns:
            str: Formatted record.
        """
        levelname = record.levelname
        original_msg = record.getMessage()
        logger_name = record.name

        # Add color to level name
        colored_levelname = (
            f"{self.COLORS.get(levelname, '')}{levelname}{Style.RESET_ALL}"
        )

        return f"{colored_levelname} ({logger_name}) :      {original_msg}"


def configure_logging() -> logging.Logger:
    """Configure logger for the library.

    Returns:
        logging.Logger: the configured logger.
    """
    library_logger = logging.getLogger(ROOT_LOGGER_NAME)
    library_logger.setLevel(LOG_LEVEL)

    # setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    formatter = ColoredFormatter()
    console_handler.setFormatter(formatter)

    # add handler to library logger
    library_logger.addHandler(console_handler)

    return library_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get logger, configuring on first use."""
    global _logger_configured  # noqa: PLW0603

    if not _logger_configured:
        configure_logging()
        _logger_configured = True

    # Return specific logger
    logger_name = f"{ROOT_LOGGER_NAME}.{name}" if name else ROOT_LOGGER_NAME
    return logging.getLogger(logger_name)
