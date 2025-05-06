from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
@app.post("/predict/manual")
async def predict_manual(data: SensorData):
    global latest_iot_data
    try:
        input_array = np.array([[data.voltage, data.current, data.temperature, data.power, data.vibration]])
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array).max() if hasattr(model, "predict_proba") else 0.5

        latest_iot_data = data.dict()  # Store for IoT use

        return {
            "prediction": str(prediction),
            "probability": round(float(probability), 2)
        }
    except Exception as e:
        traceback.print_exc()
        return {"prediction": "Error", "probability": 0.0, "error": str(e)}

# Simulated GET prediction
@app.get("/predict/auto")
async def predict_auto():
    try:
        data = {
            "voltage": round(random.uniform(100, 250), 2),
            "current": round(random.uniform(0.1, 10), 2),
            "temperature": round(random.uniform(10, 100), 2),
            "vibration": round(random.uniform(0.0, 1.0), 2)
        }
        data["power"] = round(data["voltage"] * data["current"], 2)

        input_array = np.array([[data["voltage"], data["current"], data["temperature"], data["power"], data["vibration"]]])
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array).max() if hasattr(model, "predict_proba") else 0.5

        return {
            "prediction": str(prediction),
            "probability": round(float(probability), 2)
        }
    except Exception as e:
        traceback.print_exc()
        return {"prediction": "Error", "probability": 0.0, "error": str(e)}

# IoT or manual last stored prediction
@app.get("/predict/iot")
async def predict_iot():
    global latest_iot_data
    try:
        if model is None:
            raise RuntimeError("Model not loaded.")

        if latest_iot_data is None:
            return JSONResponse(status_code=204, content={"message": "No IoT/manual data received yet."})

        data = latest_iot_data
        print(f"📡 Using stored IoT/manual data: {data}")
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
        print("❌ Error during predict_iot:")
        traceback.print_exc()
        return {
            "prediction": "Error",
            "probability": 0.0,
            "error": str(e)
        }
