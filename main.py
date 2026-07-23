from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

from src.data.dataset import prepare_dataset
from src.evaluation.evaluator import Evaluator
from src.evaluation.sklearn import SklearnEvaluationStrategy
from src.features.engineering import IdentityFeatureEngineer
from src.features.pipeline import FeatureEngineeringPipeline
from src.ml_preprocessing.preprocessor import preprocess_dataset
from src.split.splitter import split_dataset
from src.training.sklearn import SklearnTrainingStrategy
from src.training.trainer import Trainer

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]
PREPROCESSING_CONFIG = {
    "numeric_features": {
        "transformer": "numeric",
        "columns": [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
        ],
    },
    "categorical_features": {
        "transformer": "categorical",
        "columns": [
            "gender",
            "SeniorCitizen",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
        ],
    },
}

def main() -> None:
    df = prepare_dataset(
        "WA_Fn-UseC_-Telco-Customer-Churn.csv", required_columns=REQUIRED_COLUMNS
    )

    samp = IdentityFeatureEngineer()
    feature_eng = FeatureEngineeringPipeline(engineers=[samp])
    df = feature_eng.transform(df)

    splitted = split_dataset(df, "Churn")
    processed_data = preprocess_dataset(dataset=splitted, config=PREPROCESSING_CONFIG)

    model = LogisticRegression(class_weight="balanced")

    train_strategy = SklearnTrainingStrategy(model)
    trainer = Trainer(train_strategy)
    train_result = trainer.train(data=processed_data)

    eval_strategy = SklearnEvaluationStrategy([recall_score,precision_score, accuracy_score, f1_score])
    evaluator = Evaluator(strategy=eval_strategy)
    eval_result = evaluator.evaluate(model=model, processed_data=processed_data)

    print(df.head())
    print(train_result)
    print(eval_result)


if __name__ == "__main__":
    main()
