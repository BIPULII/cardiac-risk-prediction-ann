import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load the saved model and scaler
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('cardiac_ann_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

st.title("Cardiac Disease Risk Prediction System")
st.write("Enter patient clinical parameters below to assess risk using our ANN.")

# UI Inputs (Match these exactly to the 13 UCI dataset features Piyumi prepares)
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", value=120)
chol = st.number_input("Serum Cholestoral in mg/dl", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
restecg = st.selectbox("Resting ECG results (0-2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.number_input("ST depression induced by exercise", value=1.0)
slope = st.selectbox("Slope of the peak exercise ST segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

if st.button("Predict Cardiac Risk"):
    # 1. Gather inputs into a numpy array
    patient_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                              thalach, exang, oldpeak, slope, ca, thal]])
    
    # 2. Scale the data using the saved scaler
    patient_data_scaled = scaler.transform(patient_data)
    
    # 3. Predict using the ANN
    prediction = model.predict(patient_data_scaled)
    risk_probability = prediction[0][0]
    
    # 4. Display Results
    if risk_probability > 0.5:
        st.error(f"⚠️ High Risk of Cardiac Disease Detected. (Confidence: {risk_probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Cardiac Disease. (Confidence: {1 - risk_probability:.2%})")