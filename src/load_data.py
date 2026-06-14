import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL, echo=False)


def load_raw_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Fix TotalCharges — has empty strings for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Convert Churn Yes/No to 1/0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Lowercase all column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    missing = df.isna().sum().sum()
    print(f"Missing values after cleaning: {missing}")
    return df


def load_to_postgres(df: pd.DataFrame) -> None:

    # Table 1: customers
    customers_df = df[[
        "customerid", "gender", "seniorcitizen", "partner",
        "dependents", "tenure", "phoneservice",
        "internetservice", "contract", "paperlessbilling"
    ]].copy()

    customers_df.columns = [
        "customer_id", "gender", "senior_citizen", "partner",
        "dependents", "tenure_months", "phone_service",
        "internet_service", "contract_type", "paperless_billing"
    ]
    customers_df["signup_date"] = None
    customers_df["city"] = None

    # Table 2: subscriptions
    subs_df = df[[
        "customerid", "monthlycharges", "totalcharges",
        "paymentmethod", "multiplelines", "onlinesecurity",
        "onlinebackup", "deviceprotection", "techsupport",
        "streamingtv", "streamingmovies"
    ]].copy()

    subs_df.columns = [
        "customer_id", "monthly_charges", "total_charges",
        "payment_method", "multiple_lines", "online_security",
        "online_backup", "device_protection", "tech_support",
        "streaming_tv", "streaming_movies"
    ]

    # Table 3: churn_labels
    labels_df = df[["customerid", "churn"]].copy()
    labels_df.columns = ["customer_id", "churned"]
    labels_df["churn_date"] = None
    labels_df["reason"] = None

    # Load to PostgreSQL
    with engine.begin() as conn:
        customers_df.to_sql("customers",    conn,
                            if_exists="replace", index=False, method="multi")
        subs_df.to_sql("subscriptions",     conn,
                       if_exists="replace", index=False, method="multi")
        labels_df.to_sql("churn_labels",    conn,
                         if_exists="replace", index=False, method="multi")

    # Verify row counts
    print("\n✓ Data loaded to PostgreSQL:")
    with engine.connect() as conn:
        for table in ["customers", "subscriptions", "churn_labels"]:
            count = conn.execute(
                text(f"SELECT COUNT(*) FROM {table}")
            ).scalar()
            print(f"  {table}: {count:,} rows")


if __name__ == "__main__":
    RAW_PATH = "data/raw/telco_churn.csv"
    df = load_raw_data(RAW_PATH)
    df = clean_data(df)
    load_to_postgres(df)
    print("\nPhase 2 complete — data is in PostgreSQL!")