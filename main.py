import os
import pickle
import warnings
import io
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Suppress sklearn/xgboost version warnings during deserialization
warnings.filterwarnings("ignore")

# Initialize FastAPI App
app = FastAPI(
    title="Chronic Kidney Disease (CKD) Diagnostic API",
    description="High-performance Clinical Decision Support REST API powered by XGBoost & WHO Rule-Based Expert Logic.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mount static files and templates
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load Trained Model
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_ckd_model.pkl")
model = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("[OK] XGBoost CKD Pipeline successfully loaded.")
except Exception as e:
    print(f"[WARN] Error loading model: {e}")

# Import rule-based logic
try:
    from rule_based import rule_based_ckd_predict
except ImportError:
    def rule_based_ckd_predict(data):
        return "Rule-based logic module not loaded"

FEATURE_NAMES = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
    'bu', 'sc', 'sod', 'pot', 'hemo', 'wc', 'htn', 'dm',
    'cad', 'appet', 'pe', 'ane'
]

# Pydantic Patient Schema
class PatientData(BaseModel):
    age: float = Field(48.0, ge=1, le=120, description="Age in years")
    bp: float = Field(80.0, ge=40, le=250, description="Blood Pressure in mm/Hg")
    sg: float = Field(1.020, ge=1.000, le=1.040, description="Specific Gravity")
    al: int = Field(0, ge=0, le=5, description="Albumin (0 to 5)")
    su: int = Field(0, ge=0, le=5, description="Sugar (0 to 5)")
    rbc: int = Field(0, ge=0, le=1, description="Red Blood Cells: 0=Normal, 1=Abnormal")
    pc: int = Field(0, ge=0, le=1, description="Pus Cells: 0=Normal, 1=Abnormal")
    pcc: int = Field(0, ge=0, le=1, description="Pus Cell Clumps: 0=Not Present, 1=Present")
    ba: int = Field(0, ge=0, le=1, description="Bacteria: 0=Not Present, 1=Present")
    bu: float = Field(36.0, ge=1, le=500, description="Blood Urea (mg/dL)")
    sc: float = Field(1.2, ge=0.1, le=25.0, description="Serum Creatinine (mg/dL)")
    sod: float = Field(135.0, ge=50, le=200, description="Sodium (mEq/L)")
    pot: float = Field(4.5, ge=1.0, le=15.0, description="Potassium (mEq/L)")
    hemo: float = Field(15.4, ge=2.0, le=22.0, description="Hemoglobin (g/dL)")
    wc: float = Field(8000.0, ge=1000, le=50000, description="White Blood Cell Count (/cu.mm)")
    htn: int = Field(0, ge=0, le=1, description="Hypertension: 0=No, 1=Yes")
    dm: int = Field(0, ge=0, le=1, description="Diabetes Mellitus: 0=No, 1=Yes")
    cad: int = Field(0, ge=0, le=1, description="Coronary Artery Disease: 0=No, 1=Yes")
    appet: int = Field(1, ge=0, le=1, description="Appetite: 0=Poor, 1=Good")
    pe: int = Field(0, ge=0, le=1, description="Pedal Edema: 0=No, 1=Yes")
    ane: int = Field(0, ge=0, le=1, description="Anemia: 0=No, 1=Yes")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 48, "bp": 80, "sg": 1.020, "al": 1, "su": 0,
                "rbc": 0, "pc": 0, "pcc": 0, "ba": 0, "bu": 36,
                "sc": 1.2, "sod": 135, "pot": 4.5, "hemo": 15.4, "wc": 8000,
                "htn": 1, "dm": 0, "cad": 0, "appet": 1, "pe": 0, "ane": 0
            }
        }

def analyze_biomarkers(data: dict) -> List[str]:
    """Provides clinical biomarker findings for decision support."""
    insights = []
    if data.get("sc", 0) > 1.4:
        insights.append(f"Elevated Serum Creatinine ({data['sc']} mg/dL) indicating impaired glomerular filtration.")
    if data.get("bu", 0) > 40:
        insights.append(f"Elevated Blood Urea ({data['bu']} mg/dL) indicating nitrogenous waste retention.")
    if data.get("al", 0) >= 1:
        insights.append(f"Albuminuria detected (Grade {data['al']}) suggesting renal filter permeability compromise.")
    if data.get("hemo", 15) < 11.5:
        insights.append(f"Low Hemoglobin ({data['hemo']} g/dL) suggestive of secondary renal anemia.")
    if data.get("htn", 0) == 1:
        insights.append("Hypertension recorded: major risk factor accelerating nephrosclerosis.")
    if data.get("dm", 0) == 1:
        insights.append("Diabetes Mellitus recorded: major risk factor for diabetic nephropathy.")
    if data.get("sg", 1.020) < 1.015:
        insights.append(f"Low Specific Gravity ({data['sg']}) reflecting impaired tubular concentration.")
    if not insights:
        insights.append("All key physiological biomarkers fall within standard reference intervals.")
    return insights

# -------------------------------------------------------------
# Web UI Endpoint
# -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def serve_home(request: Request):
    """Serves the interactive web diagnostic portal."""
    return templates.TemplateResponse("index.html", {"request": request})

# -------------------------------------------------------------
# Health Check Endpoint
# -------------------------------------------------------------
@app.get("/health", tags=["System"])
async def health_check():
    """System health check for cloud containers and monitoring."""
    return {
        "status": "healthy",
        "service": "CKD Clinical Diagnostic API",
        "framework": "FastAPI + ASGI",
        "model_loaded": model is not None,
        "version": "2.0.0"
    }

# -------------------------------------------------------------
# Sample Data Endpoint (For 1-Click Testing)
# -------------------------------------------------------------
@app.get("/api/sample/{sample_type}", tags=["Diagnostic Services"])
async def get_sample_data(sample_type: str):
    """Returns verified clinical sample data for fast testing."""
    if sample_type.lower() == "healthy":
        return {
            "age": 30, "bp": 70, "sg": 1.025, "al": 0, "su": 0,
            "rbc": 0, "pc": 0, "pcc": 0, "ba": 0, "bu": 20,
            "sc": 0.8, "sod": 140, "pot": 4.2, "hemo": 14.0, "wc": 7500,
            "htn": 0, "dm": 0, "cad": 0, "appet": 1, "pe": 0, "ane": 0
        }
    elif sample_type.lower() == "at_risk":
        return {
            "age": 60, "bp": 150, "sg": 1.010, "al": 3, "su": 1,
            "rbc": 1, "pc": 1, "pcc": 1, "ba": 0, "bu": 120,
            "sc": 2.5, "sod": 130, "pot": 5.0, "hemo": 9.0, "wc": 22000,
            "htn": 1, "dm": 1, "cad": 0, "appet": 0, "pe": 1, "ane": 1
        }
    else:
        raise HTTPException(status_code=400, detail="Invalid sample type. Choose 'healthy' or 'at_risk'.")

# -------------------------------------------------------------
# Machine Learning Prediction Endpoint
# -------------------------------------------------------------
@app.post("/api/predict", tags=["Diagnostic Services"])
async def predict_ckd(patient: PatientData):
    """
    Predict Chronic Kidney Disease using trained XGBoost ML Pipeline.
    Returns diagnostic classification, confidence percentage, and clinical insights.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Machine Learning model is not loaded.")

    try:
        data_dict = patient.model_dump()
        input_df = pd.DataFrame([data_dict])[FEATURE_NAMES]

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        ckd_prob = float(probabilities[1])
        no_ckd_prob = float(probabilities[0])

        is_ckd = bool(prediction == 1)
        result_label = "CKD Detected" if is_ckd else "No CKD Detected"

        if ckd_prob >= 0.70:
            risk_level = "High"
        elif ckd_prob >= 0.35:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        confidence = round((ckd_prob if is_ckd else no_ckd_prob) * 100, 2)
        insights = analyze_biomarkers(data_dict)

        return {
            "status": "success",
            "prediction": result_label,
            "is_ckd": is_ckd,
            "risk_level": risk_level,
            "ckd_probability": round(ckd_prob, 4),
            "no_ckd_probability": round(no_ckd_prob, 4),
            "confidence_percentage": confidence,
            "biomarker_insights": insights,
            "patient_parameters": data_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# -------------------------------------------------------------
# Rule-Based WHO Assessment Endpoint
# -------------------------------------------------------------
@app.post("/api/rule-based", tags=["Diagnostic Services"])
async def rule_based_assessment(patient: PatientData):
    """
    Evaluates patient parameters using deterministic WHO clinical guidelines.
    """
    try:
        data_dict = patient.model_dump()
        assessment = rule_based_ckd_predict(data_dict)
        insights = analyze_biomarkers(data_dict)

        return {
            "status": "success",
            "method": "Rule-Based Expert System (WHO Guidelines)",
            "assessment": assessment,
            "biomarker_insights": insights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rule evaluation error: {str(e)}")

# -------------------------------------------------------------
# Batch CSV Prediction Endpoint
# -------------------------------------------------------------
@app.post("/api/predict/batch", tags=["Diagnostic Services"])
async def predict_batch_csv(file: UploadFile = File(...)):
    """
    Accepts CSV upload of multiple patient biomarker records and returns batch predictions.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="ML model is not initialized.")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .csv file.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Check required columns
        missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"CSV missing required columns: {missing_cols}"
            )

        processed_df = df[FEATURE_NAMES]
        preds = model.predict(processed_df)
        probs = model.predict_proba(processed_df)[:, 1]

        results = []
        for i in range(len(df)):
            is_ckd = bool(preds[i] == 1)
            results.append({
                "patient_index": i + 1,
                "prediction": "CKD Detected" if is_ckd else "No CKD",
                "is_ckd": is_ckd,
                "ckd_probability": round(float(probs[i]), 4),
                "risk_level": "High" if probs[i] >= 0.7 else ("Moderate" if probs[i] >= 0.35 else "Low")
            })

        total = len(results)
        ckd_count = sum(1 for r in results if r["is_ckd"])

        return {
            "status": "success",
            "total_records": total,
            "ckd_positive_count": ckd_count,
            "ckd_negative_count": total - ckd_count,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
