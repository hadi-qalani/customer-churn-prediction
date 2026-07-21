from sklearn.tree import DecisionTreeClassifier

from src.data.dataset import prepare_dataset
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
    df = prepare_dataset("WA_Fn-UseC_-Telco-Customer-Churn.csv", required_columns=REQUIRED_COLUMNS)

    samp = IdentityFeatureEngineer()
    feature_eng = FeatureEngineeringPipeline(engineers=[samp])
    df = feature_eng.transform(df)

    splitted = split_dataset(df, "Churn")
    processed_data = preprocess_dataset(dataset=splitted, config=PREPROCESSING_CONFIG)

    model = DecisionTreeClassifier(max_depth=30)

    strategy = SklearnTrainingStrategy(model)
    trainer = Trainer(strategy)
    result = trainer.train(data=processed_data)

    print(df.head())
    print(result)


if __name__ == "__main__":
    main()
