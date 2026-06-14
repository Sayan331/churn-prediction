import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw data from PostgreSQL into model-ready features.
    Every decision here is backed by the EDA findings.
    """
    df = df.copy()

    # ── 1. Handle missing values ─────────────────────────────────
    # 11 missing total_charges — all are new customers with tenure=0
    # They haven't been charged yet — impute with 0
    df['total_charges'] = df['total_charges'].fillna(0)

    # ── 2. Drop customer_id — not a predictive feature ───────────
    df = df.drop(columns=['customer_id'])

    # ── 3. Binary encoding — Yes/No and Male/Female columns ──────
    binary_cols = {
        'gender':             {'Male': 1, 'Female': 0},
        'partner':            {'Yes': 1, 'No': 0},
        'dependents':         {'Yes': 1, 'No': 0},
        'phone_service':      {'Yes': 1, 'No': 0},
        'paperless_billing':  {'Yes': 1, 'No': 0},
    }
    for col, mapping in binary_cols.items():
        df[col] = df[col].map(mapping)

    # ── 4. Three-value columns ────────────────────────────────────
    # These have: Yes / No / No internet service or No phone service
    # Treat 'No internet/phone service' as 0 (same as No)
    three_val_cols = [
        'multiple_lines', 'online_security', 'online_backup',
        'device_protection', 'tech_support',
        'streaming_tv', 'streaming_movies'
    ]
    for col in three_val_cols:
        df[col] = df[col].map(
            {'Yes': 1, 'No': 0,
             'No internet service': 0,
             'No phone service': 0}
        )

    # ── 5. One-hot encode nominal categoricals ────────────────────
    # contract_type: Month-to-month / One year / Two year
    # internet_service: DSL / Fiber optic / No
    # payment_method: 4 payment types
    nominal_cols = ['contract_type', 'internet_service', 'payment_method']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)

    # ── 6. New features from EDA insights ────────────────────────

    # Avg monthly spend (total / tenure) — normalises for tenure length
    # Avoid division by zero for new customers
    df['avg_monthly_spend'] = np.where(
        df['tenure_months'] > 0,
        df['total_charges'] / df['tenure_months'],
        df['monthly_charges']
    )

    # Tenure groups — EDA showed churners leave early
    df['tenure_group'] = pd.cut(
        df['tenure_months'],
        bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3],  # 0=new, 1=1yr, 2=2yr, 3=loyal
        include_lowest=True
    ).astype(int)

    # High spender flag — churned customers pay more
    high_spend_threshold = df['monthly_charges'].median()
    df['high_spender'] = (
        df['monthly_charges'] > high_spend_threshold
    ).astype(int)

    # Service count — how many add-on services does customer have
    service_cols = [
        'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies'
    ]
    df['service_count'] = df[service_cols].sum(axis=1)

    # Convert boolean columns from get_dummies to int
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def get_X_y(df: pd.DataFrame):
    """Split engineered DataFrame into features and target."""
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from src.db import get_engine, get_training_data

    engine = get_engine()
    raw_df = get_training_data(engine)

    print(f"Raw shape: {raw_df.shape}")
    engineered_df = engineer_features(raw_df)
    print(f"Engineered shape: {engineered_df.shape}")

    X, y = get_X_y(engineered_df)
    print(f"\nFeature matrix: {X.shape}")
    print(f"Target: {y.shape}")
    print(f"\nNew features created:")
    new_features = ['avg_monthly_spend', 'tenure_group',
                    'high_spender', 'service_count']
    print(X[new_features].describe().round(2))
    print(f"\nAll features:")
    print(X.columns.tolist())