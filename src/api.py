from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import tensorflow as tf

# Initialize the microservice
app = FastAPI(title="Cardiac Risk API")

# Load the AI assets into memory on startup
model = tf.keras.models.load_model('models/cardiac_ann_model.h5')
scaler = joblib.load('outputs/scaler.pkl')

# Define the exact data structure we expect from the UI
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def predict_risk(data: PatientData):
    try:
        # 1. Extract values into a numpy array
        features = np.array([[
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang,
            data.oldpeak, data.slope, data.ca, data.thal
        ]])
        
        # 2. Scale the data using Member 1's pipeline scaler
        scaled_features = scaler.transform(features)
        
        # 3. Query the Neural Network
        prediction = model.predict(scaled_features)
        risk_prob = float(prediction[0][0])
        
        # 4. Return a clean JSON response
        return {
            "risk_probability": risk_prob,
            "high_risk": bool(risk_prob > 0.5)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))