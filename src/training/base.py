from abc import ABC, abstractmethod

import pandas as pd


class TrainingStrategy(ABC):
    """Abstract interface for model training strategies.

    All training strategy implementations must provide methods for
    fitting a model and generating predictions.
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Train the underlying model.

        Args:
            X: Feature matrix used for training.
            y: Target values corresponding to ``X``.
        """

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Generate predictions for the given input data.

        Args:
            X: Feature matrix for prediction.

        Returns:
            Predicted target values.
        """