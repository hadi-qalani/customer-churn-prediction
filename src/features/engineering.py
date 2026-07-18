import pandas as pd

from .base import BaseFeatureEngineer


class IdentityFeatureEngineer(BaseFeatureEngineer):
    """Feature engineer that returns the input dataframe unchanged."""

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return a copy of the input dataframe.

        Args:
            df: Input dataframe.

        Returns:
            An unchanged copy of the input dataframe.
        """
        return df.copy()
