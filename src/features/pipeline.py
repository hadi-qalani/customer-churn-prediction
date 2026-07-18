import pandas as pd

from src.features.base import BaseFeatureEngineer


class FeatureEngineeringPipeline:
    """Apply a sequence of feature engineering steps to a dataframe.

    Attributes:
        engineers: Ordered list of feature engineers to apply.
    """

    def __init__(
        self,
        engineers: list[BaseFeatureEngineer],
    ) -> None:
        """Initialize the feature engineering pipeline.

        Args:
            engineers: Ordered list of feature engineer instances.
        """
        self.engineers = engineers

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineers sequentially.

        Args:
            df: Input dataframe.

        Returns:
            The transformed dataframe after applying all feature engineers.
        """
        transformed_df = df.copy()

        for engineer in self.engineers:
            transformed_df = engineer.transform(transformed_df)

        return transformed_df