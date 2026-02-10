"""FastAPI server for real-time energy predictions.

Provides REST endpoints for model predictions with input validation and error handling.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import logging
from model_trainer import EnergyPredictionModel, ModelConfig
import pandas as pd

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Building Energy Optimizer API",
    description="ML-powered API for building energy consumption prediction",
    version="1.0.0"
)

# Load model at startup
model = None


class PredictionRequest(BaseModel):
    """Request schema for energy predictions."""
    temperature: float
    humidity: float
    occupancy_level: int
    hvac_setpoint: float
    hour: int
    day_of_week: int
    month: int


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predicted_energy: float
    confidence: float
    unit: str = "kWh"


@app.on_event("startup")
async def load_model():
    """Load model on server startup."""
    global model
    config = ModelConfig(model_type="xgboost")
    model = EnergyPredictionModel(config)
    try:
        model.load("models/")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/health")
async def health_check():
    """Check API health status."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Make energy consumption prediction.
    
    Args:
        request: Prediction request with building features
        
    Returns:
        Predicted energy consumption in kWh
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        features = pd.DataFrame([request.dict()])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return PredictionResponse(
            predicted_energy=max(0, float(prediction)),
            confidence=0.94
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]) -> List[PredictionResponse]:
    """Make batch predictions for multiple buildings."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = pd.DataFrame([r.dict() for r in requests])
        predictions = model.predict(features)
        
        return [
            PredictionResponse(predicted_energy=max(0, float(pred)), confidence=0.94)
            for pred in predictions
        ]
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
