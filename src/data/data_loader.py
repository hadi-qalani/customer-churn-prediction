from pathlib import Path

import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a CSV dataset from the given file path.

    Args:
        file_path (str): Path to the CSV dataset file.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the file format is not CSV.
        pd.errors.EmptyDataError: If the CSV file is empty.
    """

    _file_path = Path(file_path)

    if not _file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    if _file_path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    return pd.read_csv(file_path)
