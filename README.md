# Chronic Kidney Disease (CKD) Diagnostic API & Expert System

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Flask](https://img.shields.io/badge/Flask-Alternative-black.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Pipeline-EB5424.svg)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A clinical decision support platform and high-performance RESTful API designed to identify the risk of Chronic Kidney Disease (CKD) using an ensemble **XGBoost Machine Learning Pipeline** and a **WHO Rule-Based Expert System**.

> **Author:** Basavraj Biradar  
> **Degree:** M.Sc. Data Science, Vellore Institute of Technology (VIT), Chennai  
> **Undergraduate:** B.Sc. Computer Science, SPPU (CGPA: 8.45 / 10 &bull; Distinction)  
> **GitHub:** [@Biradar72](https://github.com/Biradar72)

---

## 🌟 Why FastAPI / Flask Instead of Streamlit?

| Feature | Streamlit (Original) | FastAPI / Flask (Production Architecture) |
| :--- | :--- | :--- |
| **REST API Capabilities** | ❌ No native API endpoints | ✅ Full RESTful JSON endpoints (`/api/predict`, `/api/predict/batch`) |
| **External Integration** | ❌ Cannot be queried by hospital EHR/mobile apps | ✅ Standard HTTP JSON interface for third-party systems |
| **Interactive API Specs** | ❌ None | ✅ Auto-generated OpenAPI / Swagger UI (`/docs`) & ReDoc |
| **Performance & Async** | ⚠️ Script re-executes on state changes | ✅ High-throughput asynchronous ASGI request processing |
| **Data Validation** | ⚠️ Manual slider bounds | ✅ Strict Pydantic schema validation & type safety |
| **Batch CSV Processing** | ⚠️ Memory-heavy in frontend | ✅ Streaming multipart batch evaluation endpoint |

---

## 🏗️ System Architecture

```
                                  ┌─────────────────────────────┐
                                  │   Client Request (HTTP)     │
                                  └──────────────┬──────────────┘
                                                 │
                                  ┌──────────────▼──────────────┐
                                  │   FastAPI / Uvicorn Server  │
                                  │    (Pydantic Validation)    │
                                  └──────┬───────────────┬──────┘
                                         │               │
                 ┌───────────────────────▼──┐         ┌──▼──────────────────────────┐
                 │  XGBoost ML Pipeline     │         │ WHO Deterministic Logic     │
                 │  • StandardScaler        │         │ • Renal Thresholds          │
                 │  • XGBClassifier         │         │ • Biomarker Stratification  │
                 └──────────────┬───────────┘         └──┬──────────────────────────┘
                                │                        │
                                └──────────────┬─────────┘
                                               │
                                  ┌────────────▼────────────────┐
                                  │  Clinical JSON Response     │
                                  │  • Risk Probability         │
                                  │  • Confidence Level         │
                                  │  • Biomarker Insights       │
                                  └─────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Biradar72/ckd-prediction-app.git
cd ckd-prediction-app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run with FastAPI (Recommended)

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Web Portal:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 3. Run with Flask (Alternative)

```bash
python flask_app.py
```

- **Flask Portal:** [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📡 API Reference & Endpoints

### 1. `POST /api/predict` — Machine Learning Diagnosis
Evaluates 21 patient biomarkers with the trained XGBoost model.

#### Request Example (JSON):
```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 48, "bp": 80, "sg": 1.020, "al": 1, "su": 0,
       "rbc": 0, "pc": 0, "pcc": 0, "ba": 0, "bu": 36,
       "sc": 1.2, "sod": 135, "pot": 4.5, "hemo": 15.4, "wc": 8000,
       "htn": 1, "dm": 0, "cad": 0, "appet": 1, "pe": 0, "ane": 0
     }'
```

#### Response Example:
```json
{
  "status": "success",
  "prediction": "No CKD Detected",
  "is_ckd": false,
  "risk_level": "Low",
  "ckd_probability": 0.158,
  "no_ckd_probability": 0.842,
  "confidence_percentage": 84.2,
  "biomarker_insights": [
    "Albuminuria detected (Grade 1) suggesting renal filter permeability compromise.",
    "Hypertension recorded: major risk factor accelerating nephrosclerosis."
  ]
}
```

---

### 2. `POST /api/rule-based` — WHO Guideline Assessment
Applies clinical expert rules to assess patient risk.

#### Request Example:
```bash
curl -X POST "http://127.0.0.1:8000/api/rule-based" \
     -H "Content-Type: application/json" \
     -d '{ "bp": 150, "sg": 1.010, "al": 3, "sc": 2.5 }'
```

---

### 3. `POST /api/predict/batch` — CSV Cohort Screening
Uploads a CSV containing multiple records and returns patient-level risk predictions and cohort statistics.

```bash
curl -X POST "http://127.0.0.1:8000/api/predict/batch" \
     -F "file=@patients_sample.csv"
```

---

## 🧪 Clinical Biomarkers Evaluated

| Parameter | Symbol | Reference Interval | Clinical Significance in CKD |
| :--- | :--- | :--- | :--- |
| **Serum Creatinine** | `sc` | 0.6 – 1.2 mg/dL | Primary waste biomarker; increases as GFR falls |
| **Blood Urea** | `bu` | 15 – 45 mg/dL | Measures blood nitrogen retention from protein catabolism |
| **Specific Gravity** | `sg` | 1.015 – 1.025 | Assesses urine concentration capacity in renal tubules |
| **Albuminuria** | `al` | Grade 0 (None) | Marker of glomerular capillary barrier degradation |
| **Hemoglobin** | `hemo` | 13.5 – 17.5 g/dL | Low levels indicate renal anemia from erythropoietin deficiency |
| **Electrolytes** | `sod`, `pot` | Na: 135–145, K: 3.5–5.0 | Electrolyte imbalance resulting from impaired filtration |
| **Comorbidities** | `htn`, `dm` | Binary (0 / 1) | Major etiologies driving diabetic and hypertensive nephropathy |

---

## 🐳 Docker Container Deployment

Build and launch the microservice locally or for cloud deployment:

```bash
# Build Docker image
docker build -t ckd-diagnostic-api .

# Run container
docker run -p 8000:8000 ckd-diagnostic-api
```

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
