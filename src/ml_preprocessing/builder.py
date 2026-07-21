from sklearn.compose import ColumnTransformer
from src.ml_preprocessing.registry import PREPROCESSING_REGISTRY


def build_preprocessor(config: dict) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer from a preprocessing configuration.

    The configuration maps transformer names to their corresponding pipeline
    factories and the columns they should be applied to. Each transformer is
    instantiated using the registered factory in PREPROCESSING_REGISTRY.

    Args:
        config (dict):
            Preprocessing configuration where each key is a transformer name
            and each value is a dictionary containing:

            - transformer: Name of the registered transformer factory.
            - columns: List of column names assigned to the transformer.

    Returns:
        ColumnTransformer:
            A configured ColumnTransformer containing all preprocessing
            pipelines defined in the configuration.
    """

    transformers = []

    for name, settings in config.items():
        transformer_name = settings["transformer"]

        factory = PREPROCESSING_REGISTRY[transformer_name]

        transformer = factory()

        columns = settings["columns"]

        transformers.append(
            (
                name,
                transformer,
                columns,
            )
        )

    return ColumnTransformer(transformers=transformers)
