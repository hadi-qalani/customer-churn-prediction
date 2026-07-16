from src.data.data_loader import load_dataset


def main() -> None:
    df = load_dataset("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    print(df.head())


if __name__ == "__main__":
    main()
