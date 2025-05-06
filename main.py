from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime, timedelta
from fastapi.responses import FileResponse
import numpy as np
import random
import traceback
import os
import logging
import sys

# Set up basic logging to stdout (for Render)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,  # Use INFO or DEBUG as needed
    format="%(asctime)s - %(levelname)s - %(message)s"
)


app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
IOT_DATA_EXPIRY_MINUTES = 2  # Set how long IOT data is valid

# Load ML model
# Load your trained ML model
try:
    model = joblib.load("random_forest_model.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:")
    traceback.print_exc()
    model = None

class SensorData(BaseModel):
    voltage: float
    current: float
    temperature: float
    power: float
    vibration: float

# Store latest IoT data and timestamp
latest_iot_data = None
latest_iot_time = None

# Serve index.html from root
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

# Manual prediction (no effect on IoT state)
@app.post("/predict/manual")
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
@app.get("/predict/auto")
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

# Update IoT data (e.g., from backend device or simulator)
@app.post("/update_iot_data")
def update_iot_data(data: SensorData):
    global latest_iot_data, latest_iot_time
    logging.debug(f"🟢 IoT data received: {latest_iot_data}")

    latest_iot_data = data.dict()

    
    latest_iot_time = datetime.now()
    return {"status": "IOT data received"}

# Get prediction from the latest IoT data
@app.get("/predict/iot")
def get_iot_prediction():
    global latest_iot_data, latest_iot_time

    if latest_iot_data is None or latest_iot_time is None:
        return {"prediction": "No data", "probability": 0.0}

    # Check if the data has expired
    if datetime.now() - latest_iot_time > timedelta(minutes=IOT_DATA_EXPIRY_MINUTES):
        return {"prediction": "No recent data", "probability": 0.0}


    logging.debug(f"📊 Predicting using IoT data: {latest_iot_data}")

    # Convert the latest data to a DataFrame
    df = pd.DataFrame([latest_iot_data])

    # Rename columns to match model's training data
    df = df.rename(columns={
        "voltage": "Voltage",
        "current": "Current",
        "temperature": "Temperature",
        "power": "Power",
        "vibration": "Vibration"
    })
    
    # Make the prediction and get the probability
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    logging.debug(f"🔍 IoT Prediction: {pred}, Probability: {proba}")

    return {"prediction": int(pred), "probability": float(proba)}


