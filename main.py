from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime, timedelta
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import random
import traceback
import os
import logging
import sys
from typing import Optional
from fastapi.staticfiles import StaticFiles

# Global buffer to store partial data
latest_prediction_result = None
latest_iot_data = {}
latest_iot_time = None
required_fields = {"voltage", "current", "temperature", "power", "vibration", "humidity"}

# Set up basic logging to stdout (for Render)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
IOT_DATA_EXPIRY_SECONDS = 5  # 100 ms: Only combine readings within 100 ms

MODEL_PATH = os.path.join(os.getcwd(), 'random_forest_model.pkl')

# Load ML model
try:
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at path: {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at path: {MODEL_PATH}")

    loaded_obj = joblib.load(MODEL_PATH)
    if isinstance(loaded_obj, tuple):
        model, expected_fields = loaded_obj
    else:
        model = loaded_obj
        expected_fields = ['Voltage', 'Current', 'Temperature', 'Power', 'Vibration', 'Humidity']
    print("✅ Model loaded successfully.")

except Exception as e:
    print("❌ Failed to load model:")
    traceback.print_exc()
    model = None

class SensorData(BaseModel):
    voltage: Optional[float] = None
    current: Optional[float] = None
    temperature: Optional[float] = None
    power: Optional[float] = None
    vibration: Optional[float] = None
    humidity: Optional[float] = None

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

        input_data = pd.DataFrame([{
            "Voltage": data.voltage,
            "Current": data.current,
            "Temperature": data.temperature,
            "Power": data.power,
            "Vibration": data.vibration,
            "Humidity": data.humidity
        }])

        input_df = input_data[expected_fields]
        logging.info(f"Received input: {input_data}")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
            prediction = int(probability > 0.5)
        else:
            prediction = model.predict(input_df)[0]
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
        voltage = round(random.uniform(230, 233), 2)
        current = round(random.uniform(0.2, 0.40), 2)
        temperature = round(random.uniform(30, 45), 2)
        power = round(voltage * current, 2)
        vibration = round(random.uniform(0.0, 1.0), 2)
        humidity = round(random.uniform(35.0, 40.0), 2)

        print(f"🔄 Auto-refresh simulated input: voltage={voltage}, current={current}, temperature={temperature}, power={power}, vibration={vibration}")

        input_data = pd.DataFrame([{
            "Voltage": voltage,
            "Current": current,
            "Temperature": temperature,
            "Power": power,
            "Vibration": vibration,
            "Humidity": humidity
        }])
       
        input_df = input_data[expected_fields]

        logging.info(f"Received input: {input_data}")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
            prediction = int(probability > 0.5)
        else:
            prediction = model.predict(input_df)[0]
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

@app.get("/predict/iotLive")
async def predict_iotLive():
    try:
        logging.info("🔄 Received request to /predict/iotLive")
        if latest_prediction_result is None:
            logging.warning("⚠️ No prediction result available. Returning 204.")
            return Response(status_code=204)
        logging.info(f"✅ Returning prediction result: {latest_prediction_result}")
        return latest_prediction_result
    except Exception as e:
        logging.error("❌ Error in /predict/iotLive:", exc_info=True)
        return {
            "prediction": "Error",
            "probability": 0.0,
            "error": str(e)
        }

@app.post("/predict_iot")
def predict_iot(data: SensorData):
    global latest_iot_data, latest_iot_time, latest_prediction_result

    try:
        now = datetime.now()

        # If buffer is too old, clear it before merging new data
        if latest_iot_time is not None:
            age = (now - latest_iot_time).total_seconds()
            if age > IOT_DATA_EXPIRY_SECONDS:
                latest_iot_data.clear()
                logging.info(f"🧹 Cleared stale buffer (age={age:.3f}s)")

        # Merge incoming partial data
        incoming_data = data.dict(exclude_unset=True)
        logging.debug(f"📥 Incoming partial data: {incoming_data}")

        latest_iot_data.update(incoming_data)
        latest_iot_time = now
        logging.debug(f"🗃️ Updated latest_iot_data: {latest_iot_data}")

        # Check for missing fields
        missing = required_fields - latest_iot_data.keys()
        if missing:
            logging.debug(f"⏳ Missing fields: {missing}")
            return {"status": "Waiting for more data", "missing_fields": list(missing)}

        logging.debug(f"✅ All required fields received. Proceeding to prediction.")

        # Convert to DataFrame
        df = pd.DataFrame([latest_iot_data])
        logging.debug(f"📊 Raw DataFrame: {df}")

        input_data = df.rename(columns={
            "voltage": "Voltage",
            "current": "Current",
            "temperature": "Temperature",
            "power": "Power",
            "vibration": "Vibration",
            "humidity": "Humidity"
        })
        logging.debug(f"🔄 Renamed input data: {input_data}")

        input_df = input_data[expected_fields]
        logging.debug(f"📥 Final input for model: {input_df}")

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        prediction = 1 if probability > 0.5 else 0

        latest_prediction_result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "timestamp": datetime.now().isoformat()
        }

        logging.debug(f"🤖 Prediction: {prediction}, Probability: {probability}")

        # Clear buffer after prediction
        latest_iot_data.clear()
        logging.debug("🧹 Cleared latest_iot_data after prediction.")

        return {"prediction": int(prediction), "probability": float(probability)}

    except Exception as e:
        logging.error("❌ Error during IoT prediction:", exc_info=True)
        return {
            "prediction": "Error",
            "probability": 0.0,
            "error": str(e)
        }
