# main.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import joblib
from pydantic import BaseModel
from typing import List, Optional
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Churn Prediction API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CustomerData(BaseModel):
    Tenure: float
    NumberOfAddress: float
    CashbackAmount: float
    DaySinceLastOrder: float
    OrderCount: float
    SatisfactionScore: float

class PredictionResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    risk_level: str
    confidence: str
    threshold: float

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]
    summary: dict

# Load model
try:
    bundle = joblib.load('churn_model.pkl')
    model = bundle['pipeline']
    feature_config = bundle['feature_config']
    threshold = bundle['threshold']
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise RuntimeError("Model loading failed")

@app.get("/")
async def root():
    return {"message": "Churn Prediction API", "status": "healthy", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/model-info")
async def model_info():
    return {
        "features": feature_config['expected_features'],
        "threshold": threshold,
        "numeric_means": feature_config['numeric_means'],
        "model_type": "Calibrated XGBoost"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerData):
    try:
        # Convert to DataFrame
        input_data = customer.dict()
        input_df = pd.DataFrame([input_data])
        
        # Predict
        probability = model.predict_proba(input_df)[0, 1]
        will_churn = probability >= threshold
        
        # Risk levels
        if probability >= 0.7:
            risk_level = "critical"
            confidence = "high"
        elif probability >= 0.5:
            risk_level = "high" 
            confidence = "medium"
        elif probability >= 0.3:
            risk_level = "medium"
            confidence = "low"
        else:
            risk_level = "low"
            confidence = "very_low"
        
        return {
            "churn_probability": round(probability, 4),
            "will_churn": bool(will_churn),
            "risk_level": risk_level,
            "confidence": confidence,
            "threshold": threshold
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    try:
        # Read file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(400, "Unsupported file format")
        
        # Predict
        probabilities = model.predict_proba(df)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # Prepare results
        results = []
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            results.append({
                "customer_id": i,
                "churn_probability": round(prob, 4),
                "will_churn": bool(pred),
                "risk_level": "high" if prob >= 0.5 else "low"
            })
        
        # Summary
        churn_count = sum(predictions)
        summary = {
            "total_customers": len(df),
            "churn_count": churn_count,
            "churn_rate": round(churn_count / len(df), 4),
            "avg_probability": round(np.mean(probabilities), 4)
        }
        
        return {
            "predictions": results,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adjust-threshold")
async def adjust_threshold(new_threshold: float):
    if not 0.0 <= new_threshold <= 1.0:
        raise HTTPException(400, "Threshold must be between 0 and 1")
    
    global threshold
    threshold = new_threshold
    logger.info(f"Threshold updated to: {threshold}")
    
    return {"message": f"Threshold updated to {threshold}", "new_threshold": threshold}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)