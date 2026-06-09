import streamlit as st
import requests

# 1. UI Headers & Description
st.set_page_config(page_title="Cardiac Risk Predictor", layout="centered")
st.title("🫀 AI-Based Cardiac Disease Risk Prediction")
st.markdown("Enter patient clinical data below to assess the probability of cardiac disease risk using our highly optimized Artificial Neural Network.")

# 2. Input Fields (Arranged in clean columns)
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=55)
    sex = st.selectbox("Sex (1=M, 0=F)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3], index=2)
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=140)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=250)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120 (1=T, 0=F)", [0, 1])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (1=Y, 0=N)", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, value=1.0)

with col3:
    slope = st.selectbox("Slope of Peak ST (0-2)", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3], index=2)

# 3. Prediction Execution
if st.button("Predict Cardiac Risk", type="primary"):
    payload = {
        "features": [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    }
    
    try:
        # Note: Pointing to the Docker internal network name we will use
        response = requests.post("http://fastapi-backend:8000/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            prob = data['risk_probability']
            
            # 4. Risk Level Logic
            st.divider()
            if prob <= 0.40:
                risk_level = "Low Risk 🟢"
                st.success(f"### Prediction: No Heart Disease Risk")
            elif prob <= 0.70:
                risk_level = "Medium Risk 🟡"
                st.warning(f"### Prediction: Elevated Risk Detected")
            else:
                risk_level = "High Risk 🔴"
                st.error(f"### Prediction: Heart Disease Risk Present")
            
            st.markdown(f"**Risk Probability:** {prob:.1%}")
            st.markdown(f"**Risk Level:** {risk_level}")
            
        else:
            st.error("Error connecting to the Neural Network backend.")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")

# 5. Medical Disclaimer
st.divider()
st.caption("⚠️ **Disclaimer:** This system is developed for academic purposes only. It is not a replacement for professional medical diagnosis, advice, or treatment.")