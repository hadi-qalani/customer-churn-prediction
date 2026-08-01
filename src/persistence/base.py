from abc import ABC, abstractmethod
from pathlib import Path

from sklearn.base import BaseEstimator


class PersistenceStrategy(ABC):
    """Abstract interface for model persistence strategies.

    All persistence strategy implementations must provide methods
    for saving and loading trained machine learning models.

    This interface allows different persistence mechanisms such as
    Joblib, Pickle, or ONNX to be used interchangeably.
    """

    @abstractmethod
    def save(self, artifact: BaseEstimator, path: Path) -> None:
        """Save a trained model to persistent storage.

        Args:
            model: Trained machine learning model to save.
            path: Destination path where the model will be stored.

        """
        raise NotImplementedError

    @abstractmethod
    def load(self, path: Path) -> BaseEstimator:
        """Load a trained model from persistent storage.

        Args:
            path: Path of the stored model file.

        Returns:
            The loaded trained machine learning model.
        """
        raise NotImplementedError
