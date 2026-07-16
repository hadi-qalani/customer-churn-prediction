from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def load_dataset(filename: str) -> pd.DataFrame:
    """
    Load a dataset from the data directory.

    Parameters
    ----------
    filename : str
        CSV file name.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    file_path = DATA_DIR / filename
    return pd.read_csv(file_path)