from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator

from src.evaluation.result import EvaluationResult


class EvaluationStrategy(ABC):
    """Abstract base class for model evaluation strategies."""

    @abstractmethod
    def evaluate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> EvaluationResult:
        """Evaluates a trained model on a dataset.

        Args:
            model: The trained model to evaluate.
            X: Feature matrix used for evaluation.
            y: Ground truth target values corresponding to ``X``.

        Returns:
            An ``EvaluationResult`` containing the computed evaluation metrics.
        """
