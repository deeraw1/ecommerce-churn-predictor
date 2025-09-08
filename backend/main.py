from fastapi import FastAPI, HTTPException, UploadFile, File, Query
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pydantic import BaseModel
import traceback
import sqlite3
import json
from typing import List, Optional
import io
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
import os
import uvicorn

# ==================== PYDANTIC MODELS ====================
class CustomerData(BaseModel):
    tenure: int
    numberofaddress: int
    cashbackamount: float
    daysincelastorder: int
    ordercount: int
    satisfactionscore: int

class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    threshold_used: float
    risk_level: str
    insights: List[str]
    message: str

# ==================== DATA SANITIZER CLASS ====================
class DataSanitizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_config):
        self.feature_config = feature_config

    def fit(self, X, y=None):
        self.feature_config['numeric_means'] = {col: X[col].mean() for col in self.feature_config['numeric_features']}
        return self

    def transform(self, X):
        X = X.rename(columns=lambda x: x.strip().lower())
        expected = [col for col in self.feature_config['expected_features']]
        missing = [f for f in expected if f not in X.columns]
        for col in missing:
            X[col] = self.feature_config['numeric_means'][col]
        X = X[expected]
        for col in expected:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(self.feature_config['numeric_means'][col])
        return X

# ==================== MODEL LOADING ====================
def load_model_safely(model_path):
    """Load model with proper class definitions"""
    try:
        return joblib.load(model_path)
    except AttributeError as e:
        if "DataSanitizer" in str(e):
            print("⚠️  Model requires DataSanitizer class. Adding to global scope...")
            import __main__
            __main__.DataSanitizer = DataSanitizer
            return joblib.load(model_path)
        raise

# ==================== UTILITY FUNCTIONS ====================
def normalize_column_names(df: pd.DataFrame):
    """Normalize column names: lowercase, strip spaces, handle common variations"""
    column_mapping = {}
    
    for col in df.columns:
        normalized = str(col).strip().lower()
        
        variations = {
            'tenure': ['tenure', 'customer_tenure', 'months_active', 'subscription_length'],
            'numberofaddress': ['numberofaddress', 'address_count', 'num_addresses', 'shipping_addresses'],
            'cashbackamount': ['cashbackamount', 'cashback', 'reward_amount', 'cashback_earned'],
            'daysincelastorder': ['daysincelastorder', 'last_order_days', 'days_since_last', 'recency'],
            'ordercount': ['ordercount', 'total_orders', 'order_count', 'purchase_count'],
            'satisfactionscore': ['satisfactionscore', 'satisfaction', 'customer_score', 'rating']
        }
        
        for standard_name, aliases in variations.items():
            if normalized in aliases:
                column_mapping[col] = standard_name
                break
        else:
            column_mapping[col] = normalized
    
    return df.rename(columns=column_mapping)

def find_matching_columns(df: pd.DataFrame, required_columns: List[str]):
    """Find which required columns are present and which are missing"""
    present_columns = [col for col in required_columns if col in df.columns]
    missing_columns = [col for col in required_columns if col not in df.columns]
    return present_columns, missing_columns

# ==================== DATABASE FUNCTIONS ====================
def init_database():
    """Initialize the SQLite database with indexes"""
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tenure INTEGER,
            numberofaddress INTEGER,
            cashbackamount REAL,
            daysincelastorder INTEGER,
            ordercount INTEGER,
            satisfactionscore INTEGER,
            churn_prediction INTEGER,
            churn_probability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            input_data TEXT,
            insights TEXT
        )
    ''')
    
    # Create indexes for performance
    c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_churn ON predictions(churn_prediction)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_probability ON predictions(churn_probability)')
    
    conn.commit()
    conn.close()
    print("✅ SQLite database initialized successfully with indexes!")

def save_prediction_to_db(customer_data, prediction, probability, insights=None):
    """Save prediction results to database efficiently"""
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO predictions 
            (tenure, numberofaddress, cashbackamount, daysincelastorder, 
             ordercount, satisfactionscore, churn_prediction, churn_probability,
             input_data, insights)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            customer_data.tenure,
            customer_data.numberofaddress,
            customer_data.cashbackamount,
            customer_data.daysincelastorder,
            customer_data.ordercount,
            customer_data.satisfactionscore,
            int(prediction),
            round(float(probability), 3),
            json.dumps(customer_data.dict()),
            json.dumps(insights) if insights else None
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database save error: {e}")
        return False

def get_prediction_history(limit: int = 10):
    """Retrieve prediction history from database"""
    try:
        conn = sqlite3.connect('predictions.db')
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = c.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"❌ Database query error: {e}")
        return []

# ==================== INSIGHT GENERATION ====================
def generate_insights(customer_data, churn_probability):
    """Generate actionable insights based on prediction"""
    insights = []
    
    if churn_probability > 0.17:
        insights.append("🚨 CRITICAL RISK: Immediate retention actions needed!")
    elif churn_probability > 0.12:
        insights.append("⚠️  MEDIUM RISK: Customer showing churn signals")
    else:
        insights.append("✅ LOW RISK: Customer appears loyal")
    
    if customer_data.daysincelastorder > 30:
        insights.append("📧 Last order was over 30 days ago - send re-engagement campaign")
    
    if customer_data.satisfactionscore < 3:
        insights.append("⭐ Low satisfaction score - consider support follow-up")
    
    if customer_data.cashbackamount < 15:
        insights.append("💰 Below average cashback - offer promotion")
    
    if customer_data.ordercount < 5:
        insights.append("🛒 Low order count - recommend loyalty program")
    
    return insights

# ==================== BATCH PROCESSING ====================
def process_batch_from_dataframe(df: pd.DataFrame, filename: str):
    """Process batch prediction from DataFrame efficiently"""
    try:
        if churn_model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        print(f"📨 Processing {len(df)} customers from {filename}")
        
        # Normalize column names
        df = normalize_column_names(df)
        
        # Define required columns
        required_cols = ['tenure', 'numberofaddress', 'cashbackamount', 
                        'daysincelastorder', 'ordercount', 'satisfactionscore']
        
        # Check for missing columns
        present_cols, missing_cols = find_matching_columns(df, required_cols)
        
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}. Using default values.")
        
        # Prepare data with required columns
        df_clean = df.copy()
        for col in required_cols:
            if col not in df_clean.columns:
                if col in ['tenure', 'numberofaddress', 'daysincelastorder', 'ordercount', 'satisfactionscore']:
                    df_clean[col] = 0
                elif col == 'cashbackamount':
                    df_clean[col] = 0.0
        
        df_clean = df_clean[required_cols].fillna(0)
        
        # Batch predictions (much faster than row-by-row)
        probabilities = churn_model["pipeline"].predict_proba(df_clean)
        prob_churn = probabilities[:, 1]
        threshold = churn_model.get("threshold", 0.17)
        predictions = [1 if p >= threshold else 0 for p in prob_churn]
        
        # Process results efficiently
        results = []
        successful_saves = 0
        
        for i in range(len(df_clean)):
            row = df_clean.iloc[i]
            customer_data = CustomerData(**{k: row[k] for k in required_cols})
            insights = generate_insights(customer_data, prob_churn[i])
            
            if missing_cols:
                insights.append(f"📝 Note: Used default values for missing columns: {missing_cols}")
            
            # Save to database
            if save_prediction_to_db(customer_data, predictions[i], prob_churn[i], insights):
                successful_saves += 1
            
            # Only keep first 100 results in memory for preview
            if i < 100:
                results.append({
                    "row_id": i + 1,
                    "churn_prediction": int(predictions[i]),
                    "churn_probability": round(float(prob_churn[i]), 3),
                    "risk_level": "high" if prob_churn[i] > 0.17 else "medium" if prob_churn[i] > 0.12 else "low",
                    "insights": insights
                })
        
        print(f"💾 Successfully saved {successful_saves}/{len(df_clean)} predictions to database")
        
        # Calculate summary statistics
        churn_count = sum(predictions)
        retention_count = len(predictions) - churn_count
        churn_rate = churn_count / len(predictions) if len(predictions) > 0 else 0
        avg_probability = np.mean(prob_churn) if len(prob_churn) > 0 else 0
        
        return {
            "filename": filename,
            "total_customers": len(df_clean),
            "saved_to_db": successful_saves,
            "data_quality": {
                "missing_columns": missing_cols,
                "note": "Default values used for missing columns" if missing_cols else "All columns present"
            },
            "predictions": results,
            "summary": {
                "churn_count": churn_count,
                "retention_count": retention_count,
                "churn_rate": round(float(churn_rate), 3),
                "average_probability": round(float(avg_probability), 3)
            },
            "note": "Only showing first 100 predictions. All data saved to database."
        }
        
    except Exception as e:
        error_msg = f"Batch processing failed: {str(e)}"
        print(f"❌ Error: {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

# ==================== LOAD MODEL & INITIALIZE ====================
try:
    model_path = "C:\\Users\\USER\\Documents\\github_repo\\ecommerce-churn-predictor\\model\\churn_model.pkl"
    churn_model = load_model_safely(model_path)
    print("✅ Model loaded successfully!")
    print(f"📊 Model features: {churn_model['model_info']['features_used']}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(traceback.format_exc())
    churn_model = None

# Initialize database
init_database()

# ==================== FASTAPI APP ====================
app = FastAPI(title="E-commerce Churn Predictor API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
frontend_dir = "frontend"

if os.path.exists(os.path.join(frontend_dir, "css")):
    app.mount("/css", StaticFiles(directory=os.path.join(frontend_dir, "css")), name="css")

if os.path.exists(os.path.join(frontend_dir, "js")):
    app.mount("/js", StaticFiles(directory=os.path.join(frontend_dir, "js")), name="js")


# Create sample CSV if it doesn't exist
def create_sample_csv():
    sample_data = """tenure,numberofaddress,cashbackamount,daysincelastorder,ordercount,satisfactionscore
12,2,25.5,15,8,4
24,1,45.2,5,15,5
6,3,12.8,45,3,2
18,2,32.1,8,12,4
36,1,67.8,2,25,5"""
    
    with open('sample.csv', 'w') as f:
        f.write(sample_data)
    print("✅ Sample CSV created")

if not os.path.exists('sample.csv'):
    create_sample_csv()

# ==================== API ENDPOINTS ====================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page"""
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            return f.read()
    return """
    <html>
        <head><title>E-commerce Churn Predictor</title></head>
        <body>
            <h1>E-commerce Churn Predictor API</h1>
            <p>Frontend files not found. Please ensure the 'frontend' directory exists.</p>
            <p>API endpoints:</p>
            <ul>
                <li><a href="/health">/health</a> - Health check</li>
                <li><a href="/predictions/history">/predictions/history</a> - Prediction history</li>
                <li><a href="/sample.csv">/sample.csv</a> - Sample CSV file</li>
            </ul>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if churn_model is None:
        return {"status": "error", "message": "Model not loaded"}
    else:
        return {
            "status": "healthy", 
            "message": "Model loaded successfully",
            "features": churn_model["model_info"]["features_used"],
            "database": "SQLite (predictions.db)",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/predictions/history")
async def get_history(limit: int = Query(10, ge=1, le=1000)):
    """Get recent prediction history"""
    try:
        history = get_prediction_history(limit)
        return {
            "count": len(history),
            "predictions": history,
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(data: CustomerData):
    """Single prediction endpoint"""
    try:
        if churn_model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        print(f"📨 Received single prediction request")
        
        input_data = pd.DataFrame([{
            'tenure': data.tenure,
            'numberofaddress': data.numberofaddress,
            'cashbackamount': data.cashbackamount,
            'daysincelastorder': data.daysincelastorder,
            'ordercount': data.ordercount,
            'satisfactionscore': data.satisfactionscore
        }])
        
        probabilities = churn_model["pipeline"].predict_proba(input_data)[0]
        probability = probabilities[1]
        threshold = churn_model.get("threshold", 0.17)
        prediction = 1 if probability >= threshold else 0
        
        insights = generate_insights(data, probability)
        save_prediction_to_db(data, prediction, probability, insights)
        
        print(f"🎯 Prediction: {prediction}, Probability: {probability:.3f}")

        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=round(float(probability), 3),
            threshold_used=round(float(threshold), 3),
            risk_level="high" if probability > 0.17 else "medium" if probability > 0.12 else "low",
            insights=insights,
            message="Prediction saved to database!"
        )
        
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"❌ Error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/predict/upload/csv")
async def predict_churn_csv(file: UploadFile = File(...)):
    """Process batch predictions from CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        print(f"📨 Processing CSV file: {file.filename}")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"📊 Loaded {len(df)} rows from CSV")
        
        result = process_batch_from_dataframe(df, file.filename)
        return result
        
    except Exception as e:
        error_msg = f"CSV processing failed: {str(e)}"
        print(f"❌ Error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/predict/upload/excel")
async def predict_churn_excel(file: UploadFile = File(...)):
    """Process batch predictions from Excel file"""
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="File must be Excel format")
        
        print(f"📨 Processing Excel file: {file.filename}")
        
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        print(f"📊 Loaded {len(df)} rows from Excel")
        
        result = process_batch_from_dataframe(df, file.filename)
        return result
        
    except Exception as e:
        error_msg = f"Excel processing failed: {str(e)}"
        print(f"❌ Error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/sample.csv")
async def get_sample_csv():
    """Serve sample CSV file"""
    return FileResponse("sample.csv", media_type="text/csv", filename="sample.csv")

# Explicit OPTIONS handlers for CORS preflight requests
@app.options("/predict")
async def options_predict():
    return JSONResponse(content={"message": "OK"})

@app.options("/predict/upload/csv")
async def options_predict_csv():
    return JSONResponse(content={"message": "OK"})

@app.options("/predict/upload/excel")
async def options_predict_excel():
    return JSONResponse(content={"message": "OK"})

# ==================== STARTUP EVENT ====================
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting E-commerce Churn Predictor API...")
    print("📍 Frontend: http://localhost:8000")
    print("📍 API Health: http://localhost:8000/health")
    print("📍 Sample CSV: http://localhost:8000/sample.csv")
    print("-" * 50)
    
    if churn_model:
        print(f"✅ Model loaded with {len(churn_model['model_info']['features_used'])} features")
    else:
        print("❌ Model failed to load")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)