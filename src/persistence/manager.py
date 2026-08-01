from pathlib import Path

from sklearn.base import BaseEstimator

from src.persistence.base import PersistenceStrategy


class PersistenceManager:
    """Manager for handling model persistence operations.

    This class delegates saving and loading operations to a specific
    persistence strategy implementation.

    Attributes:
        strategy: Persistence strategy used for saving and loading objects.
    """

    def __init__(self, strategy: PersistenceStrategy) -> None:
        """Initialize PersistenceManager.

        Args:
            strategy: Concrete persistence strategy that implements
                save and load operations.
        """
        self.strategy = strategy

    def save(self, artifact: BaseEstimator, path: Path) -> None:
        """Save a model using the configured persistence strategy.

        Args:
            model: Trained machine learning model to be saved.
            path: Destination path where the model will be stored.
        """
        self.strategy.save(artifact, path)

    def load(self, path: Path) -> BaseEstimator:
        """Load a model using the configured persistence strategy.

        Args:
            path: Path of the stored model file.

        Returns:
            Loaded machine learning model.
        """
        return self.strategy.load(path)
