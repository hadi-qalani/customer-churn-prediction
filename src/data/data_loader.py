from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the data directory.

    Parameters
    ----------
    filename : str
        Name of the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the dataset does not exist.

    ValueError
        If the file is not a CSV.

    pd.errors.EmptyDataError
        If the CSV file is empty.
    """

    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            "Only CSV files are supported."
        )

    return pd.read_csv(file_path)