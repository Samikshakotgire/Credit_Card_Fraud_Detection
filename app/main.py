import pickle
import numpy as np
import time
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using XGBoost + MLOps pipeline",
    version="1.0.0"
)

# Load model and feature names at startup (not per request)
base_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_dir, "fraud_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(base_dir, "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

print(f"Model loaded. Expects {len(feature_names)} features.")


# --- Request schema ---
class TransactionRequest(BaseModel):
    """
    All V1-V28 are PCA-transformed features from the original dataset.
    Amount_log = log1p(transaction amount)
    Hour       = hour of day (0-23) extracted from Time
    """
    V1: float; V2: float; V3: float; V4: float
    V5: float; V6: float; V7: float; V8: float
    V9: float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    Amount_log: float = Field(..., description="log1p(transaction amount)")
    Hour: float       = Field(..., ge=0, le=23, description="Hour of day 0-23")

    class Config:
        json_schema_extra = {
            "example": {
                "V1": -1.3598071336738, "V2": -0.0727811733098497,
                "V3": 2.53634673796914, "V4": 1.37815522427443,
                "V5": -0.338320769942518, "V6": 0.462387777762292,
                "V7": 0.239598554061257, "V8": 0.0986979012610507,
                "V9": 0.363786969611213, "V10": 0.0907941719789316,
                "V11": -0.551599533260813, "V12": -0.617800855762348,
                "V13": -0.991389847235408, "V14": -0.311169353699879,
                "V15": 1.46817697209427, "V16": -0.470400525259478,
                "V17": 0.207971241929242, "V18": 0.0257905801985591,
                "V19": 0.403992960255733, "V20": 0.251412098239705,
                "V21": -0.018306777944153, "V22": 0.277837575558899,
                "V23": -0.110473910188767, "V24": 0.0669280749146731,
                "V25": 0.128539358273528, "V26": -0.189114843888824,
                "V27": 0.133558376740387, "V28": -0.0210530534538215,
                "Amount_log": 4.52, "Hour": 14
            }
        }


# --- Response schema ---
class PredictionResponse(BaseModel):
    prediction: str        # "FRAUD" or "LEGITIMATE"
    fraud_probability: float # 0.0 to 1.0
    confidence: str         # "HIGH" / "MEDIUM" / "LOW"
    latency_ms: float        # inference time in milliseconds
    model_version: str


# --- Endpoints ---

@app.get("/", response_class=HTMLResponse)
def root():
    try:
        index_path = os.path.join(base_dir, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        return HTMLResponse(
            content=f"<h1>Error loading index.html</h1><p>{str(e)}</p>",
            status_code=500
        )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "n_features": len(feature_names)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest):
    try:
        start = time.time()

        # Build feature vector in correct order
        feature_values = [getattr(transaction, feat) for feat in feature_names]
        X = np.array(feature_values).reshape(1, -1)

        # Predict
        fraud_prob = float(model.predict_proba(X)[0][1])
        prediction = "FRAUD" if fraud_prob > 0.5 else "LEGITIMATE"

        # Confidence band
        if fraud_prob > 0.85 or fraud_prob < 0.15:
            confidence = "HIGH"
        elif fraud_prob > 0.65 or fraud_prob < 0.35:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        latency_ms = round((time.time() - start) * 1000, 2)

        return PredictionResponse(
            prediction=prediction,
            fraud_probability=round(fraud_prob, 4),
            confidence=confidence,
            latency_ms=latency_ms,
            model_version="xgboost-optuna-v1"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(transactions: list[TransactionRequest]):
    """Score multiple transactions in one call."""
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Max 100 transactions per batch")

    results = []
    for txn in transactions:
        feature_values = [getattr(txn, feat) for feat in feature_names]
        X = np.array(feature_values).reshape(1, -1)
        prob = float(model.predict_proba(X)[0][1])
        results.append({
            "prediction": "FRAUD" if prob > 0.5 else "LEGITIMATE",
            "fraud_probability": round(prob, 4)
        })
    return {"results": results, "count": len(results)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)