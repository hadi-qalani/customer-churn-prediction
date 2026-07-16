from src.data.data_loader import load_dataset
from src.data.data_validation import validate_dataset

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


def main() -> None:
    df = load_dataset("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    validate_dataset(df, required_columns=REQUIRED_COLUMNS)

    print(df.head())


if __name__ == "__main__":
    main()
