from __future__ import annotations

import pandas as pd


def convert_column_type(
    df: pd.DataFrame,
    column: str,
    dtype: str,
    *,
    errors: str = "raise",
    true_values: list | None = None,
    false_values: list | None = None,
) -> pd.DataFrame:
    
    """
    Return a copy of the DataFrame with the specified column converted.

    Args:
        df (pd.DataFrame):
            Input DataFrame.
        column (str):
            Name of the column to convert.
        dtype (str):
            Target data type. Supported values are
            ``"numeric"``, ``"category"``, and ``"bool"``.
        errors (str):
            Error handling strategy passed to ``pd.to_numeric()``.
            Ignored for non-numeric conversions.
        true_values (list | None):
            Values that should be mapped to ``True`` when
            ``dtype="bool"``.
        false_values (list | None):
            Values that should be mapped to ``False`` when
            ``dtype="bool"``.

    Returns:
        pd.DataFrame:
            A copy of the DataFrame with the converted column.

    Raises:
        KeyError:
            If the specified column does not exist.
        ValueError:
            If ``dtype="bool"`` and either ``true_values`` or
            ``false_values`` is not provided.
        NotImplementedError:
            If the requested data type is not supported.
    """

    if column not in df.columns:
        raise KeyError(f"Column '{column}' does not exist.")

    converted_df = df.copy()

    if dtype == "numeric":
        converted_df[column] = pd.to_numeric(
            converted_df[column],
            errors=errors,
        )

    elif dtype == "category":
        converted_df[column] = converted_df[column].astype("string")

    elif dtype == "bool":
        if true_values is None or false_values is None:
            raise ValueError(
                "true_values and false_values must be provided for bool dtype."
            )

        mapping = {
            **{v: True for v in true_values},
            **{v: False for v in false_values},
        }

        converted_df[column] = converted_df[column].map(mapping).astype("boolean")

    else:
        raise NotImplementedError(f"Unsupported dtype: {dtype}")

    return converted_df