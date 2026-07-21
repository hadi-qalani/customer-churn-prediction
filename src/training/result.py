from dataclasses import dataclass

from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class TrainingResult:
    """Represents the outcome of a completed training process.

    Attributes:
        model: The trained machine learning model.
        training_time: Training duration in seconds.
    """

    model: BaseEstimator
    training_time: float
