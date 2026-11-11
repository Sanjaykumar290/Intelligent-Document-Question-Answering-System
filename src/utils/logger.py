# Logging setup
"""Logging setup shared by every module."""

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name.

    Usage in any file:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Parsed %d pages", n_pages)
    """
    logger = logging.getLogger(name)

    # Only attach a handler once. Without this check, calling get_logger()
    # repeatedly (Streamlit re-runs your script on every interaction)
    # would print each message multiple times.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # stop messages being duplicated by the root logger

    return logger