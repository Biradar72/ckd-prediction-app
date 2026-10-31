import os
import pickle
import warnings
import io
import pandas as pd
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# Load Trained Model
MODEL_PATH = os.path.join(BASE_DIR, "xgboost_ckd_model.pkl")
model = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("[OK] XGBoost CKD Pipeline successfully loaded for Flask.")
except Exception as e:
    print(f"[WARN] Error loading model: {e}")

from rule_based import rule_based_ckd_predict

FEATURE_NAMES = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
    'bu', 'sc', 'sod', 'pot', 'hemo', 'wc', 'htn', 'dm',
    'cad', 'appet', 'pe', 'ane'
]

def analyze_biomarkers(data: dict):
    insights = []
    if float(data.get("sc", 0)) > 1.4:
        insights.append(f"Elevated Serum Creatinine ({data['sc']} mg/dL) indicating impaired glomerular filtration.")
    if float(data.get("bu", 0)) > 40:
        insights.append(f"Elevated Blood Urea ({data['bu']} mg/dL) indicating nitrogenous waste retention.")
    if int(data.get("al", 0)) >= 1:
        insights.append(f"Albuminuria detected (Grade {data['al']}) suggesting renal filter permeability compromise.")
    if float(data.get("hemo", 15)) < 11.5:
        insights.append(f"Low Hemoglobin ({data['hemo']} g/dL) suggestive of secondary renal anemia.")
    if int(data.get("htn", 0)) == 1:
        insights.append("Hypertension recorded: major risk factor accelerating nephrosclerosis.")
    if int(data.get("dm", 0)) == 1:
        insights.append("Diabetes Mellitus recorded: major risk factor for diabetic nephropathy.")
    if float(data.get("sg", 1.020)) < 1.015:
        insights.append(f"Low Specific Gravity ({data['sg']}) reflecting impaired tubular concentration.")
    if not insights:
        insights.append("All key physiological biomarkers fall within standard reference intervals.")
    return insights

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "service": "CKD Clinical Diagnostic API (Flask)",
        "model_loaded": model is not None,
        "version": "2.0.0"
    })

@app.route("/api/sample/<sample_type>")
def sample(sample_type):
    if sample_type.lower() == "healthy":
        return jsonify({
            "age": 30, "bp": 70, "sg": 1.025, "al": 0, "su": 0,
            "rbc": 0, "pc": 0, "pcc": 0, "ba": 0, "bu": 20,
            "sc": 0.8, "sod": 140, "pot": 4.2, "hemo": 14.0, "wc": 7500,
            "htn": 0, "dm": 0, "cad": 0, "appet": 1, "pe": 0, "ane": 0
        })
    elif sample_type.lower() == "at_risk":
        return jsonify({
            "age": 60, "bp": 150, "sg": 1.010, "al": 3, "su": 1,
            "rbc": 1, "pc": 1, "pcc": 1, "ba": 0, "bu": 120,
            "sc": 2.5, "sod": 130, "pot": 5.0, "hemo": 9.0, "wc": 22000,
            "htn": 1, "dm": 1, "cad": 0, "appet": 0, "pe": 1, "ane": 1
        })
    return jsonify({"error": "Invalid sample type"}), 400

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])[FEATURE_NAMES]

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        ckd_prob = float(probabilities[1])
        no_ckd_prob = float(probabilities[0])
        is_ckd = bool(prediction == 1)

        risk_level = "High" if ckd_prob >= 0.70 else ("Moderate" if ckd_prob >= 0.35 else "Low")
        confidence = round((ckd_prob if is_ckd else no_ckd_prob) * 100, 2)
        insights = analyze_biomarkers(data)

        return jsonify({
            "status": "success",
            "prediction": "CKD Detected" if is_ckd else "No CKD Detected",
            "is_ckd": is_ckd,
            "risk_level": risk_level,
            "ckd_probability": round(ckd_prob, 4),
            "no_ckd_probability": round(no_ckd_prob, 4),
            "confidence_percentage": confidence,
            "biomarker_insights": insights,
            "patient_parameters": data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/rule-based", methods=["POST"])
def rule_based():
    try:
        data = request.get_json(force=True)
        assessment = rule_based_ckd_predict(data)
        insights = analyze_biomarkers(data)
        return jsonify({
            "status": "success",
            "method": "Rule-Based Expert System (WHO Guidelines)",
            "assessment": assessment,
            "biomarker_insights": insights
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/batch", methods=["POST"])
def batch_predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "File must be CSV"}), 400

    try:
        df = pd.read_csv(file)
        missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing columns: {missing_cols}"}), 400

        processed = df[FEATURE_NAMES]
        preds = model.predict(processed)
        probs = model.predict_proba(processed)[:, 1]

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

        return jsonify({
            "status": "success",
            "total_records": len(results),
            "ckd_positive_count": sum(1 for r in results if r["is_ckd"]),
            "results": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
