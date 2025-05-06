from fastapi import FastAPI, Query
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store latest IoT POSTed data
latest_iot_data = None

# Load ML model
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

# Manual POST prediction
@app.post("/predict/")
async def predict(data: SensorData):
    global latest_iot_data
    try:
        print(f"📥 Received manual data: {data.dict()}")
        if model is None:
            raise RuntimeError("Model not loaded.")

        input_array = np.array([[data.voltage, data.current, data.temperature, data.power, data.vibration]])
        prediction = model.predict(input_array)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_array).max()
        else:
            probability = 0.5

        # Save for future graphing in "iot-graph" mode
        latest_iot_data = data.dict()

        result = {
            "prediction": str(prediction),
            "probability": round(float(probability), 2)
        }
        print(f"✅ Prediction result: {result}")
        return result

    except Exception as e:
        print("❌ Error during prediction:")
        traceback.print_exc()
        return {
            "prediction": "Error",
            "probability": 0.0,
            "error": str(e)
        }

# Graph/Auto-refresh prediction
@app.get("/get_prediction")
async def get_prediction(source: str = Query("simulated")):
    global latest_iot_data
    try:
        if model is None:
            raise RuntimeError("Model not loaded.")

        if source == "iot":  # Use previously submitted data (desktop or real IoT)
            if latest_iot_data is None:
                raise RuntimeError("No IoT/manual data received yet.")
            data = latest_iot_data
            print(f"📡 Returning stored IoT/manual data: {data}")
        else:  # Simulated auto-refresh
            data = {
                "voltage": round(random.uniform(100, 250), 2),
                "current": round(random.uniform(0.1, 10), 2),
                "temperature": round(random.uniform(10, 100), 2),
                "vibration": round(random.uniform(0.0, 1.0), 2)
            }
            data["power"] = round(data["voltage"] * data["current"], 2)
            print(f"🔄 Simulated auto data: {data}")

        input_array = np.array([[data["voltage"], data["current"], data["temperature"], data["power"], data["vibration"]]])
        prediction = model.predict(input_array)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_array).max()
        else:
            probability = 0.5

        result = {
            "prediction": str(prediction),
            "probability": round(float(probability), 2)
        }
        return result

    except Exception as e:
        print("❌ Error during get_prediction:")
        traceback.print_exc()
        return {
            "prediction": "Error",
            "probability": 0.0,
            "error": str(e)
        }
