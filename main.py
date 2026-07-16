from src.data.dataset import prepare_dataset

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
    df = prepare_dataset(
        "WA_Fn-UseC_-Telco-Customer-Churn.csv", required_columns=REQUIRED_COLUMNS
    )

    print(df.head())


if __name__ == "__main__":
    main()
