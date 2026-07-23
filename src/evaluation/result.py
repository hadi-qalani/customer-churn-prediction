from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    """Stores the outcome of a model evaluation.

    Attributes:
        metrics: A mapping of metric names to their computed values.
    """

    metrics: Mapping[str, float]
