import logging


def configure_logging(*,level: str = "INFO") -> None:
    """Configure the application's logging system."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
