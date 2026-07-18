from abc import ABC, abstractmethod

import pandas as pd


class BaseFeatureEngineer(ABC):
    """Base class for all feature engineering components."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataframe."""
        raise NotImplementedError
