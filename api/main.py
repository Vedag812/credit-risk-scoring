"""
main.py — FastAPI REST API for Credit Risk Scoring
====================================================

Endpoints:
    POST /predict  — Score a loan application (returns PD, score, SHAP)
    GET  /health   — Health check for monitoring

Run:
    uvicorn api.main:app --reload --port 8000
    Then visit: http://localhost:8000/docs (Swagger UI)

INTERVIEW TALKING POINT:
    "I built a REST API so the credit risk model can be called by any
    system — the loan origination system, a mobile app, or a web portal.
    FastAPI auto-generates Swagger docs, which makes it easy for other
    teams to integrate."
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from api.schemas import LoanApplication, RiskPrediction
from src.data_loader import load_config
from src.feature_engineering import (
    create_financial_ratios,
    create_risk_buckets,
    create_interaction_features,
    encode_categoricals,
)
from src.scorecard import probability_to_score, get_risk_category

# ---- Initialize app ----
app = FastAPI(
    title="Credit Risk Scoring API",
    description="Production-grade credit risk model with SHAP explainability",
    version="1.0.0",
)

# ---- Load model and artifacts at startup ----
config = load_config()

try:
    model = joblib.load(config["paths"]["model_file"])
    feature_names = joblib.load(config["paths"]["feature_names_file"])
    
    # Try loading SHAP explainer
    try:
        shap_explainer = joblib.load(config["paths"]["shap_explainer_file"])
    except Exception:
        shap_explainer = None
        print("[WARN] SHAP explainer not found — explanations will be unavailable")
    
    print("[API] Model and artifacts loaded successfully")
    model_loaded = True
except Exception as e:
    print(f"[API] Model not found: {e}")
    print("[API] Run 'python -m src.pipeline' first to train the model")
    model_loaded = False
    model = None
    feature_names = None
    shap_explainer = None


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring / load balancers."""
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "version": "1.0.0",
    }


@app.post("/predict", response_model=RiskPrediction)
def predict_risk(application: LoanApplication):
    """Score a loan application and return risk assessment.
    
    Returns probability of default, credit score, risk category,
    approve/deny decision, and SHAP-based top risk factors.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python -m src.pipeline' first.",
        )
    
    # Convert input to DataFrame (same format as training data)
    input_data = pd.DataFrame([application.model_dump()])
    
    # Apply the same feature engineering as training
    try:
        input_data = create_financial_ratios(input_data)
        input_data = create_risk_buckets(input_data)
        
        interaction_pairs = config["feature_engineering"]["interaction_pairs"]
        input_data = create_interaction_features(input_data, interaction_pairs)
        input_data = encode_categoricals(input_data)
        
        # Align columns with training features
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[feature_names]
        
        # Replace any NaN/Inf
        input_data = input_data.replace([np.inf, -np.inf], np.nan).fillna(0)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {e}")
    
    # Get prediction
    probability = float(model.predict_proba(input_data)[:, 1][0])
    credit_score = probability_to_score(probability, config)
    risk_category = get_risk_category(credit_score)
    
    # Decision rule (can be customized)
    decision = "APPROVE" if credit_score >= 650 else "DENY"
    
    # SHAP explanation (top 5 risk factors)
    top_risk_factors = {}
    if shap_explainer is not None:
        try:
            shap_values = shap_explainer.shap_values(input_data)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]
            
            contributions = dict(zip(feature_names, sv))
            sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_risk_factors = {k: round(float(v), 4) for k, v in sorted_contrib[:5]}
        except Exception:
            top_risk_factors = {"note": 0.0}
    
    return RiskPrediction(
        probability_of_default=round(probability, 4),
        credit_score=credit_score,
        risk_category=risk_category,
        decision=decision,
        top_risk_factors=top_risk_factors,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
