from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    """Container for train and test dataset splits.

    Attributes:
        X_train: Training feature set.
        X_test: Test feature set.
        y_train: Training target values.
        y_test: Test target values.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_dataset(
    df: pd.DataFrame,
    target: str,
    *,
    test_size: float = 0.2,
    random_state: int | None = 42,
    shuffle: bool = True,
    stratify: bool = True,
) -> DatasetSplit:
    """Split a dataset into training and test sets.

    The target column is separated from the feature columns before
    splitting. Optionally, stratified sampling can be applied to
    preserve the target class distribution.

    Args:
        df: Input dataset containing both features and the target column.
        target: Name of the target column.
        test_size: Proportion of the dataset to include in the test split.
            Must be between 0 and 1.
        random_state: Seed used to ensure reproducible splits. If ``None``,
            the split is randomized each time.
        shuffle: Whether to shuffle the data before splitting.
        stratify: Whether to preserve the target class distribution in the
            train and test sets.

    Returns:
        A ``DatasetSplit`` object containing:
            - ``X_train``: Training features.
            - ``X_test``: Test features.
            - ``y_train``: Training target values.
            - ``y_test``: Test target values.

    Raises:
        ValueError: If the input DataFrame is empty.
        ValueError: If the target column does not exist.
        ValueError: If ``test_size`` is not between 0 and 1.
        ValueError: If ``stratify=True`` while ``shuffle=False``.
    """
    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if stratify and not shuffle:
        raise ValueError("stratify=True requires shuffle=True")

    X = df.drop(columns=[target])
    y = df[target]

    stratify_column = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_column,
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
