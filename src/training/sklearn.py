import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from src.training.base import TrainingStrategy


class SklearnTrainingStrategy(TrainingStrategy):
    """Training strategy for scikit-learn estimators."""

    def __init__(
        self,
        model: BaseEstimator,
    ) -> None:
        """Initialize the training strategy.

        Args:
            model: A scikit-learn estimator implementing the standard
                ``fit`` and ``predict`` interface.
        """
        self.model = model

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> BaseEstimator:
        """Train the underlying estimator.

        Args:
            X: Feature matrix used for training.
            y: Target values corresponding to ``X``.

        Returns:
            trained model
        """
        model = self.model.fit(X, y)
        return model

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Generate predictions using the underlying estimator.

        Args:
            X: Feature matrix for prediction.

        Returns:
            Predicted target values.
        """
        return self.model.predict(X)
