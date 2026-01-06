import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 1. PAGE SETUP
st.set_page_config(page_title="VPNP Infotech | Heart AI", page_icon="❤️", layout="wide")

# Custom CSS to make labels readable and buttons big
st.markdown("""
<style>
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; font-size: 18px; }
    .stNumberInput label { font-weight: bold; font-size: 14px; }
    .stSelectbox label { font-weight: bold; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

st.title("🏥 Intelligent Heart Disease Prediction System")
st.markdown("**Developed by: VPNP Infotech** | *Medical AI Diagnostic Tool*")

# 2. DATA LOADING (Auto-Training)
if not os.path.exists('heart.csv'):
    st.error("🚨 ERROR: 'heart.csv' file is missing!")
    st.warning("👉 Please drag and drop 'heart.csv' into the Files sidebar on the left.")
    st.stop()

try:
    df = pd.read_csv('heart.csv')
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Save column names to fix the warning
    feature_names = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_scaled, y)
    
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# 3. PATIENT INPUTS (With Validations & Labels)
st.write("---")
st.subheader("📝 Enter Patient Vitals")

# We use columns for layout, but NO FORM (Fixes the loading glitch)
c1, c2, c3 = st.columns(3)

with c1:
    # AGE: Validated 1-120
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=50, help="Patient's age in years.")
    
    # SEX: 1=Male, 0=Female 
    sex_txt = st.selectbox("Gender", ["Male", "Female"], help="1=Male, 0=Female")
    sex = 1 if sex_txt == "Male" else 0
    
    # CP: 0-3 
    cp_txt = st.selectbox("Chest Pain Type", 
                          ["Typical Angina (Heart Pain)", "Atypical Angina", "Non-anginal (Not Heart)", "Asymptomatic (No Pain)"],
                          help="Type of chest pain reported.")
    cp = ["Typical Angina (Heart Pain)", "Atypical Angina", "Non-anginal (Not Heart)", "Asymptomatic (No Pain)"].index(cp_txt)
    
    # TRESTBPS: 50-250
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120, help="High BP strains the heart.")

with c2:
    # CHOL: 100-600
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, help="High cholesterol blocks arteries.")
    
    # FBS: 0 or 1 
    fbs_txt = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"], help="Diabetes indicator.")
    fbs = 1 if fbs_txt == "Yes" else 0
    
    # RESTECG: 0-2 
    restecg_txt = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg_txt)
    
    # THALACH: 60-220
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150, help="Heart rate during exercise.")

with c3:
    # EXANG: 0 or 1 
    exang_txt = st.selectbox("Exercise Induced Angina?", ["No", "Yes"], help="Does chest hurt specifically when exercising?")
    exang = 1 if exang_txt == "Yes" else 0
    
    # OLDPEAK: 0.0 - 10.0
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="ECG reading measuring heart stress.")
    
    # SLOPE: 0-2
    slope_txt = st.selectbox("Slope of Peak Exercise", ["Upsloping", "Flat", "Downsloping"])
    slope = ["Upsloping", "Flat", "Downsloping"].index(slope_txt)
    
    # CA: 0-3 
    ca = st.slider("Major Vessels Colored by Fluoroscopy (0-3)", 0, 3, 0, help="0 means blocked vessels.")
    
    # THAL: 1-3 
    thal_txt = st.selectbox("Thalassemia", ["Normal", "Fixed Defect (Permanent)", "Reversible Defect (Temporary)"])
    thal = ["Normal", "Fixed Defect (Permanent)", "Reversible Defect (Temporary)"].index(thal_txt) + 1

st.write("---")

# 4. PREDICTION BUTTON
if st.button("🚀 ANALYZE RISK NOW"):
    # Create DataFrame with correct column names
    input_df = pd.DataFrame(
        [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], 
        columns=feature_names
    )
    
    # Scale inputs
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ **HIGH RISK DETECTED**")
        st.write(f"Confidence Score: **{probability * 100:.2f}%**")
        st.warning("Recommendation: Please consult a cardiologist immediately.")
    else:
        st.success(f"✅ **PATIENT IS HEALTHY**")
        st.write(f"Confidence Score: **{(1 - probability) * 100:.2f}%**")
        st.info("Recommendation: Maintain a healthy lifestyle.")
