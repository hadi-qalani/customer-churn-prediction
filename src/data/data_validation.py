import pandas as pd




def validate_dataset(df: pd.DataFrame, required_columns: list[str]) -> None:
    """
    Validate the input dataset structure.

    This function checks whether the provided DataFrame:
    - Is not empty.
    - Contains the expected columns in the correct structure.

    Args:
        df (pd.DataFrame): Input dataset that needs to be validated.
        expected_columns (list[str]): the columns the dataset should have.

    Raises:
        ValueError: If the DataFrame is empty or does not match
            the expected dataset structure.

    Returns:
        None
    """

    if df.empty:
        raise ValueError("Dataset is empty")

    if df.columns.has_duplicates:
        raise ValueError("Dataset contains duplicate column names")

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
