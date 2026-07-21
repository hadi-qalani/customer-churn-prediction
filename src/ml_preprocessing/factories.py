from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


def create_numeric_pipeline() -> Pipeline:
    """
    Create a preprocessing pipeline for numeric features.

    The pipeline performs the following steps:

    1. Imputes missing values using the median of each feature.
    2. Standardizes features to have zero mean and unit variance.

    Returns:
        Pipeline:
            A scikit-learn pipeline configured for preprocessing
            numeric features.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


def create_categorical_pipeline() -> Pipeline:
    """
    Create a preprocessing pipeline for categorical features.

    The pipeline performs the following steps:

    1. Imputes missing values using the most frequent category.
    2. Applies one-hot encoding to categorical features.
       Unknown categories encountered during inference are ignored.

    Returns:
        Pipeline:
            A scikit-learn pipeline configured for preprocessing
            categorical features.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )
