from collections.abc import Callable, Sequence

import pandas as pd
from sklearn.base import BaseEstimator

from src.evaluation.base import EvaluationStrategy
from src.evaluation.result import EvaluationResult


class SklearnEvaluationStrategy(EvaluationStrategy):
    """Evaluation strategy backed by scikit-learn metrics."""

    def __init__(
        self,
        metrics: Sequence[Callable],
    ) -> None:
        """Initialize the evaluation strategy.

        Args:
            metrics: A sequence of metric functions used to evaluate model
                predictions. Each metric must accept ``y_true`` and ``y_pred``
                as inputs and return a numeric score.
        """
        self.metrics = tuple(metrics)

    def evaluate(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> EvaluationResult:
        """Evaluate a trained model using the configured metrics.

        Args:
            model: A trained scikit-learn estimator.
            X: Feature matrix used for evaluation.
            y: Ground-truth target values corresponding to ``X``.

        Returns:
            An ``EvaluationResult`` containing the computed metric scores.
        """
        y_pred = model.predict(X)

        results: dict[str, float] = {}

        for metric in self.metrics:
            results[metric.__name__] = metric(y, y_pred)

        return EvaluationResult(metrics=results)
