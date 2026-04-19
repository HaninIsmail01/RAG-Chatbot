import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.
    Usage: logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # avoid duplicate handlers if called multiple times

    logger.setLevel(logging.DEBUG)

    # Console handler 
    # Handles logging in the console using the required format and log level.
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(_get_formatter())

    # File handler 
    # Handles logging to a file using the required format and log level.
    file_handler = logging.FileHandler("app.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_get_formatter())

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def _get_formatter() -> logging.Formatter:
    """
    Returns a configured logging Formatter with the required trace format and date format.
    The format string is "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    and the date format is "%Y-%m-%d %H:%M:%S".
    """
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )