from sklearn.base import BaseEstimator

from src.evaluation.base import EvaluationStrategy
from src.evaluation.result import EvaluationResult
from src.ml_preprocessing.preprocessor import ProcessedDataset


class Evaluator:
    """Orchestrates model evaluation using a configured strategy."""

    def __init__(self, strategy: EvaluationStrategy) -> None:
        """Initialize the evaluator.

        Args:
            strategy: The evaluation strategy used to evaluate trained models.
        """
        self.strategy = strategy

    def evaluate(
        self,
        model: BaseEstimator,
        processed_data: ProcessedDataset,
    ) -> EvaluationResult:
        """Evaluate a trained model.

        Args:
            model: A trained scikit-learn estimator.
            X: Feature matrix used for evaluation.
            y: Ground-truth target values corresponding to ``X``.

        Returns:
            An ``EvaluationResult`` containing the computed evaluation metrics.
        """
        X = processed_data.X_test
        y =  processed_data.y_test


        return self.strategy.evaluate(model, X, y)

