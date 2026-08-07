import joblib
import pandas as pd

from src.config.config_loader import load_configs
from src.data.dataset import prepare_dataset
from src.logging.config import configure_logging
from src.logging.logger import get_logger
from src.tracking.metadata import save_metadata
from src.tracking.versioning import create_version_dir


def predict() -> None:
    """Run the complete machine learning training pipeline."""

    version_dir = create_version_dir("data/inference/output")
    result_path = version_dir / "output.csv"
    metadata_path = version_dir / "metadata.json"

    configs = load_configs()

    configure_logging(level=configs["logging"]["level"])
    logger = get_logger(__name__)

    logger.info("prediction started")

    df = prepare_dataset(
        configs["prediction"]["input_path"],
        required_columns=configs["prediction"]["required_columns"],
        schema=configs["prediction"]["schema"],
    )

    pipeline = joblib.load(configs["prediction"]["pipeline_path"])
    result = pipeline.predict(df)

    logger.info("prediction completed: %s", result)

    result = pd.DataFrame(result, columns=["result"])
    result.to_csv(result_path, index=True)

    model = pipeline.named_steps["model"]

    metadata = {
        "artifact_type": "prediction",
        "model_name": type(model).__name__,
        "version": f"{version_dir}",
        "dataset": configs["prediction"]["input_path"],
    }
    save_metadata(metadata, metadata_path)

    logger.info("prediction finished")


if __name__ == "__main__":
    predict()
