from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()


def get_engine():
    """Return a SQLAlchemy engine using credentials from .env"""
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return create_engine(url, pool_pre_ping=True)


def query_to_df(sql: str, engine=None) -> pd.DataFrame:
    """Run a SQL query and return results as a DataFrame."""
    if engine is None:
        engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def get_training_data(engine=None) -> pd.DataFrame:
    """
    Joins all 3 tables into one flat DataFrame ready for ML.
    This is what a real DS does — write a JOIN query to get training data.
    """
    sql = """
        SELECT
            c.customer_id,
            c.gender,
            c.senior_citizen,
            c.partner,
            c.dependents,
            c.tenure_months,
            c.phone_service,
            c.internet_service,
            c.contract_type,
            c.paperless_billing,
            s.monthly_charges,
            s.total_charges,
            s.payment_method,
            s.multiple_lines,
            s.online_security,
            s.online_backup,
            s.device_protection,
            s.tech_support,
            s.streaming_tv,
            s.streaming_movies,
            l.churned AS target
        FROM customers c
        JOIN subscriptions s ON c.customer_id = s.customer_id
        JOIN churn_labels  l ON c.customer_id = l.customer_id
    """
    return query_to_df(sql, engine)


def get_churn_rate_by_contract(engine=None) -> pd.DataFrame:
    """Churn rate by contract type — first real business insight."""
    sql = """
        SELECT
            c.contract_type,
            COUNT(*)                        AS total_customers,
            SUM(l.churned)                  AS churned_count,
            ROUND(AVG(l.churned) * 100, 2)  AS churn_rate_pct
        FROM customers c
        JOIN churn_labels l ON c.customer_id = l.customer_id
        GROUP BY c.contract_type
        ORDER BY churn_rate_pct DESC
    """
    return query_to_df(sql, engine)


if __name__ == "__main__":
    engine = get_engine()

    print("Testing database connection...")
    df = get_training_data(engine)
    print(f"Training data shape: {df.shape}")
    print(f"\nChurn rate: {df['target'].mean() * 100:.1f}%")
    print("\nChurn rate by contract type:")
    print(get_churn_rate_by_contract(engine))