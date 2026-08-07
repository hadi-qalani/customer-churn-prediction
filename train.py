from sklearn.pipeline import Pipeline

from src.config.config_loader import load_configs
from src.data.dataset import prepare_dataset
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import load_metrics
from src.evaluation.sklearn import SklearnEvaluationStrategy
from src.features.engineering import IdentityFeatureEngineer
from src.features.pipeline import FeatureEngineeringPipeline
from src.logging.config import configure_logging
from src.logging.logger import get_logger
from src.ml_preprocessing.preprocessor import preprocess_dataset
from src.model.factory import create_model
from src.persistence.joblib import JoblibPersistence
from src.persistence.manager import PersistenceManager
from src.split.splitter import split_dataset
from src.tracking.metadata import save_metadata
from src.tracking.versioning import create_version_dir
from src.training.sklearn import SklearnTrainingStrategy
from src.training.trainer import Trainer


def train() -> None:
    """Run the complete machine learning training pipeline."""

    configs = load_configs()

    configure_logging(level=configs["logging"]["level"])
    logger = get_logger(__name__)

    logger.info("training started")

    df = prepare_dataset(
        configs["dataset"]["path"],
        required_columns=configs["dataset"]["required_columns"],
        schema=configs["schema"],
    )

    feature_pipeline = FeatureEngineeringPipeline(engineers=[IdentityFeatureEngineer()])

    df = feature_pipeline.transform(df)

    splitted = split_dataset(df, configs["dataset"]["target"])

    processed_data = preprocess_dataset(
        dataset=splitted, config=configs["preprocessing"]
    )

    model = create_model(configs["model"]["name"], configs["model"]["params"])

    trainer = Trainer(strategy=SklearnTrainingStrategy(model))

    train_result = trainer.train(data=processed_data)

    logger.info("training completed: %s", train_result)

    metrics = load_metrics(configs["evaluation"]["metrics"])

    evaluator = Evaluator(strategy=SklearnEvaluationStrategy(metrics))

    eval_result = evaluator.evaluate(
        model=train_result.model, processed_data=processed_data
    )

    logger.info("evaluation completed: %s", eval_result)

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", feature_pipeline),
            ("preprocessor", processed_data.preprocessor),
            ("model", train_result.model),
        ]
    )

    persistence = PersistenceManager(strategy=JoblibPersistence())

    model_name = type(model).__name__
    version_dir = create_version_dir("artifacts/models")
    pipeline_path = version_dir / f"{model_name}"
    metadata_path = version_dir / "metadata.json"

    persistence.save(artifact=pipeline, path=pipeline_path)

    metadata = {
        "artifact_type": "pipeline",
        "model_name": model_name,
        "version": f"{version_dir}",
        "dataset": configs["dataset"]["path"],
        "metrics": f"{eval_result}",
        "feature_engineers": feature_pipeline.engineer_names,
    }
    save_metadata(metadata, metadata_path)

    logger.info("pipeline saved: %s", version_dir)

    logger.info("training finished")


if __name__ == "__main__":
    train()
