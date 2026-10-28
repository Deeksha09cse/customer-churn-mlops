
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using a trained ML pipeline.",
    version="1.0.0",
)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return {
        "prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
    }
