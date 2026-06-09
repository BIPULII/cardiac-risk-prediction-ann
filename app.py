import streamlit as st
import requests

st.title("Cardiac Disease Risk Prediction Dashboard")
st.write("Enter patient clinical parameters below to securely query the AI microservice.")

# UI Inputs 
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120.0)
chol = st.number_input("Serum Cholestoral in mg/dl", value=200.0)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", value=150.0)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST depression induced by exercise", value=1.0)
slope = st.selectbox("Slope of the peak exercise ST segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

if st.button("Predict Cardiac Risk"):
    # 1. Package the UI data
    payload = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
        "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    
    # 2. Send it to the FastAPI Backend
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            risk_prob = result["risk_probability"]
            
            if result["high_risk"]:
                st.error(f"⚠️ High Risk of Cardiac Disease Detected. (Confidence: {risk_prob:.2%})")
            else:
                st.success(f"✅ Low Risk of Cardiac Disease. (Confidence: {1 - risk_prob:.2%})")
        else:
            st.error("Error: The API rejected the request.")
    except Exception as e:
        st.error("Connection failed. Is the FastAPI backend running?")