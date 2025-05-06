from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import joblib
import random
import traceback
import os

app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained ML model
try:
    model = joblib.load("random_forest_model.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:")
    traceback.print_exc()
    model = None

# Input schema
class SensorData(BaseModel):
    voltage: float
    current: float
    temperature: float
    power: float
    vibration: float

# Serve index.html from root
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

# Manual prediction endpoint
@app.post("/predict/")
async def predict(data: SensorData):
    try:
        print(f"📥 Received data: {data.dict()}")
        if model is None:
            raise RuntimeError("Model not loaded.")

        input_array = np.array([[data.voltage, data.current, data.temperature, data.power, data.vibration]])
        prediction = model.predict(input_array)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_array).max()
        else:
            probability = 0.5

        result = {
            "prediction": str(prediction),
            "probability": round(float(probability), 2)
        }
        print(f"✅ Prediction: {result}")
        return result

    except Exception as e:
        print("❌ Error during prediction:")
        traceback.print_exc()
        return {
            "prediction": "Error",
            "probability": 0.0,
            "error": str(e)
        }

# Auto-refresh (IoT) prediction endpoint
@app.get("/get_prediction")
async def get_prediction():
    try:
        if model is None:
            raise RuntimeError("Model not loaded.")

        # Simulate IoT sensor data
        voltage = round(random.uniform(100, 250), 2)
        current = round(random.uniform(0.1, 10), 2)
        temperature = round(random.uniform(10, 100), 2)
        power = round(voltage * current, 2)
        vibration = round(random.uniform(0.0, 1.0), 2)

        print(f"🔄 Auto-refresh simulated input: voltage={voltage}, current={current}, temperature={temperature}, power={power}, vibration={vibration}")

        input_array = np.array([[voltage, current, temperature, power, vibration]])
        prediction = model.predict(input_array)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_array).max()
        else:
            probability = 0.5

        result = {
            "prediction": str(prediction),
            "probability": round(float(probability), 2)
        }
        print(f"📡 Auto-prediction result: {result}")
        return result

    except Exception as e:
        print("❌ Error during auto-refresh prediction:")
        traceback.print_exc()
        return {
            "prediction": "Error",
            "probability": 0.0,
            "error": str(e)
        }
