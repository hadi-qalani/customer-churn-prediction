import joblib
import pandas as pd

from src.config.config_loader import load_configs
from src.features.engineering import IdentityFeatureEngineer
from src.features.pipeline import FeatureEngineeringPipeline
from src.logging.config import configure_logging
from src.logging.logger import get_logger


def predict() -> None:
    """Run the complete machine learning training pipeline."""

    configs = load_configs()

    configure_logging(level=configs["logging"]["level"])
    logger = get_logger(__name__)

    logger.info("prediction started")

    df = pd.read_csv(configs["prediction"]["input_path"])

    feature_pipeline = FeatureEngineeringPipeline(engineers=[IdentityFeatureEngineer()])

    df = feature_pipeline.transform(df)

    pipeline = joblib.load(configs["prediction"]["pipeline_path"])

    result = pipeline.predict(df)
    logger.info(result)
    result = pd.DataFrame(result, columns=["result"])

    result.to_csv(configs["prediction"]["output_path"], index=False)

    logger.info("prediction finished")


if __name__ == "__main__":
    predict()
