from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import numpy as np
import sys
import os
sys.path.append('.')

from src.features import engineer_features

# ── Load model once at startup ────────────────────────────────────
MODEL_PATH = "models/best_model.pkl"

FEATURE_COLUMNS_PATH = "models/feature_columns.pkl"

try:
    pipeline = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
    print(f"✓ Feature columns loaded: {len(feature_columns)} features")
except FileNotFoundError as e:
    raise RuntimeError(f"Model files not found. Run src/train.py first. Error: {e}")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts probability of customer churn using Logistic Regression",
    version="1.0.0"
)


# ── Request schema ────────────────────────────────────────────────
class CustomerData(BaseModel):
    customer_id:       str   = Field(example="7590-VHVEG")
    gender:            Literal['Male', 'Female']
    senior_citizen:    int   = Field(ge=0, le=1, example=0)
    partner:           Literal['Yes', 'No']
    dependents:        Literal['Yes', 'No']
    tenure_months:     int   = Field(ge=0, example=12)
    phone_service:     Literal['Yes', 'No']
    internet_service:  Literal['DSL', 'Fiber optic', 'No']
    contract_type:     Literal['Month-to-month', 'One year', 'Two year']
    paperless_billing: Literal['Yes', 'No']
    monthly_charges:   float = Field(ge=0, example=65.0)
    total_charges:     float = Field(ge=0, example=780.0)
    payment_method:    Literal[
        'Bank transfer (automatic)',
        'Credit card (automatic)',
        'Electronic check',
        'Mailed check'
    ]
    multiple_lines:    Literal['Yes', 'No', 'No phone service']
    online_security:   Literal['Yes', 'No', 'No internet service']
    online_backup:     Literal['Yes', 'No', 'No internet service']
    device_protection: Literal['Yes', 'No', 'No internet service']
    tech_support:      Literal['Yes', 'No', 'No internet service']
    streaming_tv:      Literal['Yes', 'No', 'No internet service']
    streaming_movies:  Literal['Yes', 'No', 'No internet service']


# ── Response schema ───────────────────────────────────────────────
class PredictionResponse(BaseModel):
    customer_id:       str
    churn_probability: float
    churn_prediction:  int
    risk_level:        str
    explanation:       str


# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        # Convert request to DataFrame
        data = customer.model_dump()
        customer_id = data.pop('customer_id')
        df = pd.DataFrame([data])

        # Add customer_id back for feature engineering
        df.insert(0, 'customer_id', customer_id)

        # Add target column placeholder (required by engineer_features)
        df['target'] = 0

        # Engineer features
        df_engineered = engineer_features(df)
        X = df_engineered.drop(columns=['target'])

        # Reindex to match exact training column order
        X = X.reindex(columns=feature_columns, fill_value=0)

        # Predict
        churn_prob = pipeline.predict_proba(X)[0][1]
        churn_pred = int(churn_prob >= 0.5)

        # Risk level
        if churn_prob >= 0.7:
            risk = "HIGH"
            explanation = (
                f"Customer has {churn_prob*100:.1f}% churn probability. "
                f"Immediate intervention recommended."
            )
        elif churn_prob >= 0.4:
            risk = "MEDIUM"
            explanation = (
                f"Customer has {churn_prob*100:.1f}% churn probability. "
                f"Monitor closely and consider retention offer."
            )
        else:
            risk = "LOW"
            explanation = (
                f"Customer has {churn_prob*100:.1f}% churn probability. "
                f"Customer appears stable."
            )

        return PredictionResponse(
            customer_id=customer_id,
            churn_probability=round(float(churn_prob), 4),
            churn_prediction=churn_pred,
            risk_level=risk,
            explanation=explanation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(customers: list[CustomerData]):
    """Predict churn for multiple customers at once."""
    results = []
    for customer in customers:
        result = predict(customer)
        results.append(result)
    return {"predictions": results, "count": len(results)}