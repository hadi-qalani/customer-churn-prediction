from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.compose import ColumnTransformer
from src.ml_preprocessing.builder import build_preprocessor
from src.split.splitter import DatasetSplit

ArrayLike = np.ndarray | spmatrix


@dataclass(frozen=True)
class ProcessedDataset:
    """
    Store the processed training and test datasets.

    Attributes:
        X_train (ArrayLike):
            Preprocessed training feature matrix.
        X_test (ArrayLike):
            Preprocessed test feature matrix.
        y_train (pd.Series):
            Training target values.
        y_test (pd.Series):
            Test target values.
        preprocessor (ColumnTransformer):
            Fitted preprocessing pipeline used to transform the data.
    """

    X_train: ArrayLike
    X_test: ArrayLike
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer


def preprocess_dataset(
    dataset: DatasetSplit,
    config: dict,
) -> ProcessedDataset:
    """
    Preprocess the training and test datasets.

    A preprocessing pipeline is built from the provided configuration.
    The pipeline is fitted on the training features and then applied
    to both the training and test feature sets.

    Args:
        dataset (DatasetSplit):
            Dataset containing the training and test splits.
        config (dict):
            Preprocessing configuration used to construct the
            ColumnTransformer.

    Returns:
        ProcessedDataset:
            A container holding the transformed feature matrices,
            target values, and the fitted preprocessing pipeline.
    """

    preprocessor = build_preprocessor(config)

    X_train_processed = preprocessor.fit_transform(dataset.X_train)

    X_test_processed = preprocessor.transform(dataset.X_test)

    return ProcessedDataset(
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=dataset.y_train,
        y_test=dataset.y_test,
        preprocessor=preprocessor,
    )
