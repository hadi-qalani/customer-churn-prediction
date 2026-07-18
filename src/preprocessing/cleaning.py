# src/preprocessing/cleaning.py

import pandas as pd

from src.config.config_loader import load_schema
from src.preprocessing.type_conversion import convert_column_type


def convert_dataset_types(
    df: pd.DataFrame,
    schema: dict,
) -> pd.DataFrame:

    converted_df = df.copy()

    for column, rules in schema.items():
        converted_df = convert_column_type(
            converted_df,
            column=column,
            dtype=rules["dtype"],
            errors="coerce",
            true_values=rules.get("true_values"),
            false_values=rules.get("false_values"),
        )

    return converted_df


def clean_dataset(df: pd.DataFrame, schema: dict ) -> pd.DataFrame:

    # schema = load_schema()

    cleaned_df = convert_dataset_types(df, schema)

    # more cleaning rules will be added here

    # df = validate_dataset(df, schema)
    # df = handle_missing(df, schema)
    # df = remove_duplicates(df)
    # df = handle_outliers(df, schema)

    return cleaned_df

