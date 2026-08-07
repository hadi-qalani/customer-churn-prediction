from pathlib import Path

import joblib
from sklearn.base import BaseEstimator

from src.persistence.base import PersistenceStrategy


class JoblibPersistence(PersistenceStrategy):
    """Persistence strategy using Joblib for model serialization.

    This implementation stores and loads scikit-learn compatible models
    using the Joblib serialization format.

    Joblib is optimized for objects containing large NumPy arrays and is
    commonly used for persisting machine learning models.

    Attributes:
        None.
    """

    def save(self, artifact: BaseEstimator, path: Path) -> None:
        """Save a trained model to disk using Joblib.

        Creates the parent directory structure if it does not already exist.

        Args:
            model: Trained machine learning estimator to persist.
            path: Destination file path where the model will be stored.

        Raises:
            OSError: If the model cannot be written to the specified path.
        """
        path.parent.mkdir(exist_ok=True, parents=True)

        joblib.dump(artifact, f"{path}.joblib")

    def load(self, path: Path) -> BaseEstimator:
        """Load a trained model from disk using Joblib.

        Args:
            path: File path of the persisted model.

        Returns:
            The deserialized machine learning estimator.

        Raises:
            FileNotFoundError: If the specified model file does not exist
                or is not a regular file.
        """
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        return joblib.load(path)
