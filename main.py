import os
import joblib
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Enterprise Salary Explainability Engine API",
    description="Production REST API providing salary predictions and SHAP feature attributions.",
    version="1.0.0"
)

MODEL_PATH = os.path.join("models", "salary_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model) if hasattr(model, "predict") else None
except Exception:
    model = None
    explainer = None


class SalaryInput(BaseModel):
    years_experience: float = Field(..., example=5.0)
    job_title: str = Field(..., example="Data Scientist")
    location: str = Field(..., example="Remote")
    education_level: str = Field(..., example="Master's")


@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Enterprise Salary Explainability Engine API",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict_salary(payload: SalaryInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not loaded.")

    try:
        input_df = pd.DataFrame([payload.dict()])
        prediction = float(model.predict(input_df)[0])

        explanation = {}
        if explainer is not None:
            shap_vals = explainer.shap_values(input_df)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]
            explanation = dict(zip(input_df.columns.tolist(), map(float, shap_vals[0])))

        return {
            "predicted_salary": round(prediction, 2),
            "currency": "USD",
            "shap_attributions": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")