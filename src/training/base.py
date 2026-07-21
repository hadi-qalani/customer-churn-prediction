from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.base import BaseEstimator

ArrayLike = np.ndarray | spmatrix


class TrainingStrategy(ABC):
    """Abstract interface for model training strategies.

    All training strategy implementations must provide methods for
    fitting a model and generating predictions.
    """

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: pd.Series,
    ) -> BaseEstimator:
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
