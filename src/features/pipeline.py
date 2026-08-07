import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.base import BaseFeatureEngineer


class FeatureEngineeringPipeline(
    BaseEstimator,
    TransformerMixin,
):
    """Apply a sequence of feature engineering steps to a dataframe.

    This class is compatible with scikit-learn's ``Pipeline`` API, allowing
    multiple feature engineering steps to be executed as a single transformer
    before preprocessing and model inference.

    Attributes:
        engineers: Ordered list of feature engineers applied sequentially.
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

    @property
    def engineer_names(self):
        return [type(engineer).__name__ for engineer in self.engineers]

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        """Fit the feature engineering pipeline.

        This transformer is stateless, so no fitting is required. The method
        exists only to satisfy the scikit-learn transformer interface.

        Args:
            X: Input dataframe.
            y: Optional target values. Ignored.

        Returns:
            The fitted transformer.
        """
        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply all feature engineers sequentially.

        Args:
            X: Input dataframe.

        Returns:
            The transformed dataframe after applying all feature engineering
            steps in order.
        """
        transformed = X.copy()

        for engineer in self.engineers:
            transformed = engineer.transform(transformed)

        return transformed
