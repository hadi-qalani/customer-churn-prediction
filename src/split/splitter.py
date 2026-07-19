from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetSplit:
    """Container for dataset splits."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
