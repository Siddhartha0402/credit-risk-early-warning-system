"""FastAPI backend for credit risk scoring."""

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from src.predict import predict_customer_risk


app = FastAPI(title="Credit Risk Scoring API")


@app.get("/")
def read_root():
    """Return basic API status information."""
    return {
        "api_name": "Credit Risk Scoring API",
        "status": "ok",
        "description": "Predicts customer default probability, risk band, and recommended action.",
    }


@app.post("/predict")
def predict(input_data: Dict[str, Any]):
    """Score a single customer from a JSON dictionary."""
    try:
        prediction = predict_customer_risk(input_data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "default_probability": prediction["default_probability"],
        "risk_band": prediction["risk_band"],
        "recommended_action": prediction["recommended_action"],
    }
