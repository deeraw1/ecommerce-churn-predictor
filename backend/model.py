import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "churn_model.pkl"

# Load once at startup
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
threshold = bundle["threshold"]

def predict_churn(data: pd.DataFrame):
    proba = pipeline.predict_proba(data)[:, 1]
    predictions = (proba >= threshold).astype(int)
    return predictions, proba
