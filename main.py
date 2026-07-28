from src.config.config_loader import load_configs
from src.data.dataset import prepare_dataset
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import load_metrics
from src.evaluation.sklearn import SklearnEvaluationStrategy
from src.features.engineering import IdentityFeatureEngineer
from src.features.pipeline import FeatureEngineeringPipeline
from src.ml_preprocessing.preprocessor import preprocess_dataset
from src.model.factory import create_model
from src.split.splitter import split_dataset
from src.training.sklearn import SklearnTrainingStrategy
from src.training.trainer import Trainer


def main() -> None:
    configs = load_configs()

    df = prepare_dataset(
        configs["dataset"]["name"],
        required_columns=configs["dataset"]["required_columns"],
        schema=configs["schema"],
    )

    samp = IdentityFeatureEngineer()
    feature_eng = FeatureEngineeringPipeline(engineers=[samp])
    df = feature_eng.transform(df)

    splitted = split_dataset(df, configs["dataset"]["target"])
    processed_data = preprocess_dataset(
        dataset=splitted, config=configs["preprocessing"]
    )
    model = create_model(configs["model"]["name"], configs["model"]["params"])

    train_strategy = SklearnTrainingStrategy(model)
    trainer = Trainer(train_strategy)
    train_result = trainer.train(data=processed_data)

    metrics = load_metrics(configs["evaluation"]["metrics"])

    eval_strategy = SklearnEvaluationStrategy(metrics)
    evaluator = Evaluator(strategy=eval_strategy)
    eval_result = evaluator.evaluate(model=model, processed_data=processed_data)

    print(df.head())
    print(train_result)
    print(eval_result)


if __name__ == "__main__":
    main()
