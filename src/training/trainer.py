from time import perf_counter

from src.split.splitter import DatasetSplit
from src.training.base import TrainingStrategy
from src.training.result import TrainingResult


class Trainer:
    """Coordinates the model training process using a training strategy."""

    def __init__(self, strategy: TrainingStrategy) -> None:
        """Initialize the trainer.

        Args:
            strategy: Strategy responsible for training the model.
        """
        self.strategy = strategy

    def train(self, data: DatasetSplit) -> TrainingResult:
        """Train a model and return the training result.

        Measures the training duration and returns the trained model
        together with the elapsed training time.

        Args:
            data: Dataset split containing the training features and targets.

        Returns:
            A ``TrainingResult`` containing the trained model and the
            training duration in seconds.
        """
        start = perf_counter()

        model = self.strategy.fit(data.X_train, data.y_train)

        training_time = perf_counter() - start

        return TrainingResult(
            model=model,
            training_time=training_time,
        )