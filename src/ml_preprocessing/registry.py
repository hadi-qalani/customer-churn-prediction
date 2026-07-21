from src.ml_preprocessing.factories import (
    create_categorical_pipeline,
    create_numeric_pipeline,
)

#: Registry mapping preprocessing pipeline names to their factory functions.
#:
#: Keys:
#:     - "numeric": Creates a preprocessing pipeline for numeric features.
#:     - "categorical": Creates a preprocessing pipeline for categorical features.
#:
#: Values:
#:     Factory functions that return configured scikit-learn Pipeline objects.
PREPROCESSING_REGISTRY = {
    "numeric": create_numeric_pipeline,
    "categorical": create_categorical_pipeline,
}
