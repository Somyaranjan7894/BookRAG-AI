"""Application logging configuration for BookRAG AI.

Provides consistent, sanitized, and level-configurable standard logging.
"""

import logging
import sys
from typing import Optional


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_level: Optional[str] = None) -> None:
    """Configure application-wide logging handlers and formatting.

    Args:
        log_level: Desired log level string (e.g. 'DEBUG', 'INFO', 'WARNING').
                   Defaults to INFO if not supplied or invalid.
    """
    level_name = (log_level or "INFO").upper()
    numeric_level = getattr(logging, level_name, logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Suppress overly verbose noisy logs from standard third-party libraries if necessary
    logging.getLogger("uvicorn.access").setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """Retrieve a configured logger instance with the given module name."""
    return logging.getLogger(name)
