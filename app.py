import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from rule_based import rule_based_ckd_predict


model = pickle.load(open("xgboost_ckd_model.pkl", "rb"))


st.title("Chronic Kidney Disease (CKD) Expert System")
st.write("""
Predict CKD using two expert approaches:
- **Machine Learning (XGBoost)**  
- **Rule-Based Logic (WHO Guidelines)**
""")

st.sidebar.header("User Input Section")


method = st.sidebar.radio("Choose Prediction Mode:", ["Machine Learning", "Rule-Based"])


input_df = None
if method == "Machine Learning":
    input_mode = st.sidebar.radio("Select Input Method:", ["Manual Input", "Upload CSV"])
else:
    input_mode = "Manual Input"
    st.sidebar.info("Rule-Based works only with Manual Input (CSV not supported).")


if st.sidebar.button("View Example CSV Format"):
    example_data = pd.DataFrame({
        "age": [48, 60, 30],
        "bp": [80, 150, 70],
        "sg": [1.020, 1.010, 1.025],
        "al": [1, 3, 0],
        "su": [0, 0, 0],
        "rbc": [0, 1, 0],
        "pc": [0, 1, 0],
        "pcc": [0, 1, 0],
        "ba": [0, 0, 0],
        "bu": [36, 120, 20],
        "sc": [1.2, 2.5, 0.8],
        "sod": [135, 130, 140],
        "pot": [4.5, 5.0, 4.2],
        "hemo": [15.4, 9.0, 14.0],
        "wc": [8000, 22000, 7500],
        "htn": [1, 1, 0],
        "dm": [0, 1, 0],
        "cad": [0, 0, 0],
        "appet": [1, 0, 1],
        "pe": [0, 1, 0],
        "ane": [0, 1, 0]
    })
    st.subheader("Example CSV Format")
    st.write(example_data)
    st.info("Download this as example reference. The column names must match exactly.")

# --- CSV Upload Option ---
if input_mode == "Upload CSV" and method == "Machine Learning":
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    else:
        st.sidebar.warning("Please upload a CSV file to continue.")

# --- Manual Input Section ---
if input_mode == "Manual Input" or input_df is None:
    st.sidebar.subheader("Manual Input Values")

    age = st.sidebar.slider("Age", 1, 100, 45)
    bp = st.sidebar.slider("Blood Pressure (mm Hg)", 50, 200, 120)
    sg = st.sidebar.slider("Specific Gravity", 1.000, 1.030, 1.015)
    al = st.sidebar.slider("Albumin", 0, 5, 0)
    su = st.sidebar.slider("Sugar", 0, 5, 0)
    rbc = st.sidebar.selectbox("Red Blood Cells (Normal=0, Abnormal=1)", (0, 1))
    pc = st.sidebar.selectbox("Pus Cell (Normal=0, Abnormal=1)", (0, 1))
    pcc = st.sidebar.selectbox("Pus Cell Clumps (Not Present=0, Present=1)", (0, 1))
    ba = st.sidebar.selectbox("Bacteria (Not Present=0, Present=1)", (0, 1))
    bu = st.sidebar.slider("Blood Urea (mg/dl)", 1, 400, 40)
    sc = st.sidebar.slider("Serum Creatinine (mg/dl)", 0.1, 15.0, 1.2)
    sod = st.sidebar.slider("Sodium (mEq/L)", 100, 160, 135)
    pot = st.sidebar.slider("Potassium (mEq/L)", 2, 10, 4)
    hemo = st.sidebar.slider("Hemoglobin (g/dl)", 3, 17, 13)
    wc = st.sidebar.slider("White Blood Cell Count", 2000, 30000, 8000)
    htn = st.sidebar.selectbox("Hypertension (No=0, Yes=1)", (0, 1))
    dm = st.sidebar.selectbox("Diabetes Mellitus (No=0, Yes=1)", (0, 1))
    cad = st.sidebar.selectbox("Coronary Artery Disease (No=0, Yes=1)", (0, 1))
    appet = st.sidebar.selectbox("Appetite (Good=1, Poor=0)", (0, 1))
    pe = st.sidebar.selectbox("Pedal Edema (No=0, Yes=1)", (0, 1))
    ane = st.sidebar.selectbox("Anemia (No=0, Yes=1)", (0, 1))

    input_df = pd.DataFrame([{
        "age": age, "bp": bp, "sg": sg, "al": al, "su": su,
        "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba, "bu": bu,
        "sc": sc, "sod": sod, "pot": pot, "hemo": hemo, "wc": wc,
        "htn": htn, "dm": dm, "cad": cad, "appet": appet, "pe": pe, "ane": ane
    }])


predict_clicked = st.sidebar.button("Predict")


if predict_clicked:
    st.subheader("User Input Data")
    st.write(input_df)

   
    if method == "Machine Learning":
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)[0][1]
        result = "CKD Detected" if prediction[0] == 1 else "No CKD"

        st.subheader("Machine Learning Result")
        st.write(result)
        st.write(f"Model Confidence: {prob:.2f}")

        
        labels = ["No CKD", "CKD"]
        fig, ax = plt.subplots()
        ax.bar(labels, [1 - prob, prob], color=["green", "red"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("CKD Prediction Probability (ML Model)")
        st.pyplot(fig)

    elif method == "Rule-Based":
        result = rule_based_ckd_predict(input_df.iloc[0].to_dict())
        st.subheader("Rule-Based Result")
        st.write(result)


else:
    st.subheader("About Chronic Kidney Disease (CKD)")
    st.write("""
    Chronic Kidney Disease (CKD) is a long-term condition where kidneys gradually lose their filtering ability.  
    Early detection and management can slow its progression.
    """)

    st.markdown("### How CKD Affects the Body:")
    st.markdown("""
    - Kidneys fail to remove waste → toxins build up  
    - High blood pressure due to fluid retention  
    - Weak bones and anemia  
    - Swelling in feet and hands  
    - Fatigue, nausea, and weight loss  
    """)

    st.markdown("### Prevention Tips:")
    st.markdown("""
    - Control **blood pressure** and **blood sugar**  
    - Eat a **low-salt, kidney-friendly diet**  
    - Stay hydrated but avoid over-drinking  
    - Avoid **smoking** and **alcohol**  
    - Regular **checkups** for early detection  
    """)
