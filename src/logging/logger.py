import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name.

    Args:
        name: The name of the logger, typically ``__name__``.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    return logging.getLogger(name)
