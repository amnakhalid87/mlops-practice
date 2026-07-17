"""
app.py
------
FastAPI app jo trained Linear Regression model ko load karta hai aur
/predict endpoint pe predictions serve karta hai.
"""

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "model/model.pkl"

app = FastAPI(
    title="House Price Predictor API",
    description="Simple Linear Regression model served with FastAPI",
    version="1.0.0",
)

# Model ko app startup pe ek hi baar load kar lete hain (har request pe nahi)
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class PredictionRequest(BaseModel):
    size_sqft: float = Field(..., gt=0, description="House size in square feet", json_schema_extra={"example": 1200})


class PredictionResponse(BaseModel):
    size_sqft: float
    predicted_price_lakh: float


@app.get("/")
def root():
    return {"message": "House Price Predictor API is running. See /docs for usage."}


@app.get("/health")
def health():
    """CI/CD aur monitoring ke liye simple health check."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    X = np.array([[request.size_sqft]])
    prediction = model.predict(X)[0]

    return PredictionResponse(
        size_sqft=request.size_sqft,
        predicted_price_lakh=round(float(prediction), 2),
    )
