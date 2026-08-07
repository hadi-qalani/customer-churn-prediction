import pandas as pd

from src.data.data_loader import load_dataset
from src.data.data_validation import validate_dataset
from src.preprocessing.cleaning import clean_dataset


def prepare_dataset(
    file_path: str, required_columns: list[str], schema: dict
) -> pd.DataFrame:
    """
    Load and validate a dataset.

    This function loads a dataset from the data directory and validates
    its structure before returning it.

    Args:
        file_path (str): Path of the dataset file to load.
        required_columns (list[str]): List of required column names that
            must exist in the dataset.
        schema: a dictionary that indicates the type and values of each column

    Returns:
        pd.DataFrame: The loaded and validated dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or is missing required columns.
    """
    df = load_dataset(file_path)

    validate_dataset(df, required_columns)
    df = clean_dataset(df, schema)

    return df
