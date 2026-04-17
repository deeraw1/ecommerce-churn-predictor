from fastapi import FastAPI, HTTPException, UploadFile, File
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pydantic import BaseModel
import traceback
import io
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, Response
import os
import uvicorn
from dotenv import load_dotenv

load_dotenv(".env")

# ==================== PYDANTIC MODELS ====================
class CustomerData(BaseModel):
    tenure: float
    numberofaddress: float
    cashbackamount: float
    daysincelastorder: float
    ordercount: float
    satisfactionscore: float

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    threshold_used: float
    risk_level: str

# ==================== DATA SANITIZER ====================
class DataSanitizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_config):
        self.feature_config = feature_config

    def fit(self, X, y=None):
        self.feature_config['numeric_means'] = {
            col: X[col].mean() for col in self.feature_config['numeric_features']
        }
        return self

    def transform(self, X):
        X = X.rename(columns=lambda x: x.strip().lower())
        expected = self.feature_config['expected_features']
        missing = [f for f in expected if f not in X.columns]
        for col in missing:
            X[col] = self.feature_config['numeric_means'][col]
        X = X[expected]
        for col in expected:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(
                self.feature_config['numeric_means'][col]
            )
        return X

# ==================== MODEL LOADING ====================
def load_model_safely(model_path):
    try:
        return joblib.load(model_path)
    except AttributeError as e:
        if "DataSanitizer" in str(e):
            import __main__
            __main__.DataSanitizer = DataSanitizer
            return joblib.load(model_path)
        raise

# ==================== UTILITIES ====================
COLUMN_ALIASES = {
    'tenure':            ['tenure', 'customer_tenure', 'months_active', 'subscription_length'],
    'numberofaddress':   ['numberofaddress', 'address_count', 'num_addresses', 'shipping_addresses'],
    'cashbackamount':    ['cashbackamount', 'cashback', 'reward_amount', 'cashback_earned'],
    'daysincelastorder': ['daysincelastorder', 'last_order_days', 'days_since_last', 'recency'],
    'ordercount':        ['ordercount', 'total_orders', 'order_count', 'purchase_count'],
    'satisfactionscore': ['satisfactionscore', 'satisfaction', 'customer_score', 'rating'],
}

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        norm = str(col).strip().lower()
        for std, aliases in COLUMN_ALIASES.items():
            if norm in aliases:
                mapping[col] = std
                break
        else:
            mapping[col] = norm
    return df.rename(columns=mapping)

REQUIRED_COLS = ['tenure', 'numberofaddress', 'cashbackamount',
                 'daysincelastorder', 'ordercount', 'satisfactionscore']

# ==================== LOAD MODEL ====================
churn_model = None
model_path = os.getenv("MODEL_PATH", "model/churn_model.pkl")
if os.path.exists(model_path):
    churn_model = load_model_safely(model_path)
else:
    print(f"Model file '{model_path}' not found.")

# ==================== FASTAPI APP ====================
app = FastAPI(title="E-commerce Churn Predictor API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = "frontend"
if os.path.exists(os.path.join(frontend_dir, "css")):
    app.mount("/css", StaticFiles(directory=os.path.join(frontend_dir, "css")), name="css")
if os.path.exists(os.path.join(frontend_dir, "js")):
    app.mount("/js", StaticFiles(directory=os.path.join(frontend_dir, "js")), name="js")

# ==================== ENDPOINTS ====================
@app.get("/")
async def root():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return HTMLResponse(f.read())
    return JSONResponse({"status": "ok", "version": "2.0.0"})

@app.get("/health")
async def health():
    return {
        "status": "healthy" if churn_model else "degraded",
        "model_status": "loaded" if churn_model else "not loaded",
        "features": churn_model["model_info"]["features_used"] if churn_model else None,
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(data: CustomerData):
    if churn_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        df = pd.DataFrame([data.dict()])
        proba = churn_model["pipeline"].predict_proba(df)[0]
        prob = float(proba[1])
        threshold = float(churn_model.get("threshold", 0.17))
        prediction = 1 if prob >= threshold else 0
        risk = "high" if prob > 0.17 else "medium" if prob > 0.12 else "low"
        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(prob, 3),
            threshold_used=threshold,
            risk_level=risk,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def _process_batch(df: pd.DataFrame, filename: str):
    if churn_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = normalize_column_names(df)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]

    df_clean = df.copy()
    for col in REQUIRED_COLS:
        if col not in df_clean.columns:
            df_clean[col] = 0
    df_clean = df_clean[REQUIRED_COLS].fillna(0)

    proba = churn_model["pipeline"].predict_proba(df_clean)[:, 1]
    threshold = churn_model.get("threshold", 0.17)
    preds = [1 if p >= threshold else 0 for p in proba]

    results_df = df.copy()
    results_df["Churn_Probability"] = proba
    results_df["Churn_Prediction"] = ["Churned" if p >= threshold else "Retained" for p in proba]
    results_df["Risk_Level"] = ["high" if p > 0.17 else "medium" if p > 0.12 else "low" for p in proba]
    csv_data = results_df.to_csv(index=False)

    churn_count = sum(preds)
    total = len(preds)

    preview = [
        {
            "row_id": i + 1,
            "churn_prediction": int(preds[i]),
            "churn_probability": round(float(proba[i]), 3),
            "risk_level": "high" if proba[i] > 0.17 else "medium" if proba[i] > 0.12 else "low",
        }
        for i in range(min(100, total))
    ]

    return {
        "filename": filename,
        "total_customers": total,
        "data_quality": {"missing_columns": missing},
        "predictions": preview,
        "csv_data": csv_data,
        "summary": {
            "churn_count": churn_count,
            "retention_count": total - churn_count,
            "churn_rate": round(churn_count / total, 3) if total else 0,
            "average_probability": round(float(np.mean(proba)), 3),
        },
    }

@app.post("/predict/upload/csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    return _process_batch(df, file.filename)

@app.post("/predict/upload/excel")
async def predict_excel(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="File must be Excel format")
    contents = await file.read()
    df = pd.read_excel(io.BytesIO(contents))
    return _process_batch(df, file.filename)

@app.post("/predict/upload/csv/download")
async def download_predictions(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    result = _process_batch(df, file.filename)
    return Response(
        content=result["csv_data"],
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=churn_{file.filename}"},
    )

@app.get("/sample.csv")
async def sample_csv():
    path = "sample.csv"
    if not os.path.exists(path):
        data = "tenure,numberofaddress,cashbackamount,daysincelastorder,ordercount,satisfactionscore\n"
        data += "12,2,25.5,15,8,4\n24,1,45.2,5,15,5\n6,3,12.8,45,3,2\n"
        with open(path, "w") as f:
            f.write(data)
    return FileResponse(path, media_type="text/csv", filename="sample.csv")

@app.on_event("startup")
async def startup():
    print("E-commerce Churn Predictor API v2.0 started")
    if churn_model:
        print(f"Model loaded — {len(churn_model['model_info']['features_used'])} features")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
