"""Structured logging setup (PRD Section 13: logging, not print statements)."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger writing JSON-ish structured lines to stdout.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Minimum level, e.g. ``"INFO"`` or ``"DEBUG"``.

    Returns:
        A ``logging.Logger`` with a single stream handler (idempotent setup).
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
