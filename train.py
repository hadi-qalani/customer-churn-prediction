from pathlib import Path

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
from src.training.sklearn import SklearnTrainingStrategy
from src.training.trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[0]
PIPELINE_PATH = PROJECT_ROOT / "artifacts" / "models" / "churn_pipeline.joblib"


def train() -> None:
    """Run the complete machine learning training pipeline."""

    configs = load_configs()

    configure_logging(level=configs["logging"]["level"])
    logger = get_logger(__name__)

    logger.info("training started")

    df = prepare_dataset(
        configs["dataset"]["name"],
        required_columns=configs["dataset"]["required_columns"],
        schema=configs["schema"],
    )

    feature_pipeline = FeatureEngineeringPipeline(engineers=[IdentityFeatureEngineer()])

    df = feature_pipeline.transform(df)

    splitted = split_dataset(
        df,
        configs["dataset"]["target"],
    )

    processed_data = preprocess_dataset(
        dataset=splitted,
        config=configs["preprocessing"],
    )

    model = create_model(
        configs["model"]["name"],
        configs["model"]["params"],
    )

    trainer = Trainer(strategy=SklearnTrainingStrategy(model))

    train_result = trainer.train(data=processed_data)

    logger.info(
        "training completed: %s",
        train_result,
    )

    metrics = load_metrics(configs["evaluation"]["metrics"])

    evaluator = Evaluator(strategy=SklearnEvaluationStrategy(metrics))

    eval_result = evaluator.evaluate(
        model=train_result.model,
        processed_data=processed_data,
    )

    logger.info(
        "evaluation completed: %s",
        eval_result,
    )


    pipeline = Pipeline(
        steps=[
            ("preprocessor", processed_data.preprocessor),
            ("model", train_result.model),
        ]
    )

    persistence = PersistenceManager(strategy=JoblibPersistence())

    persistence.save(
        artifact=pipeline,
        path=PIPELINE_PATH,
    )

    logger.info(
        "pipeline saved: %s",
        PIPELINE_PATH,
    )

    logger.info("training finished")


if __name__ == "__main__":
    train()
