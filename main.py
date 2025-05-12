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
from threading import Lock
from datetime import datetime, timedelta
import requests
import time

# Add these global variables at the top with other globals
iot_buffer_lock = Lock()
IOT_DATA_EXPIRY_SECONDS = 15  # Increased to 15 seconds for sensor sync

# Global state with thread safety
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

# Cache variables
cached_data = None
cache_timestamp = 0
CACHE_DURATION = 1800  # 30 minutes



# Initialize cache variables
cached_data = None
cache_timestamp = 0
CACHE_DURATION = 600  # seconds

def fetch_weather(lat, lon, api_key):
    global cached_data, cache_timestamp

    if cached_data and (time.time() - cache_timestamp) < CACHE_DURATION:
        print("Using cached weather data.")
        return cached_data

    print("Fetching new weather data from API...")
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        cached_data = data
        cache_timestamp = time.time()
        return data
    else:
        print("Error fetching weather:", data)
        return None

# ✅ Wrapper to return only humidity
def get_humidity():
    latitude = 18.5204  # Pune
    longitude = 73.8567
    api_key = "5489d93d76a18988ed2197d6795a1323"  # Use your actual key

    weather = fetch_weather(latitude, longitude, api_key)
    if weather and "main" in weather:
        return weather['main']['humidity']
    else:
        return None

# ✅ Wrapper to return only temperature
def get_temperature():
    latitude = 18.5204  # Pune
    longitude = 73.8567
    api_key = "5489d93d76a18988ed2197d6795a1323"  # Use your actual key

    weather = fetch_weather(latitude, longitude, api_key)
    if weather and "main" in weather:
        return weather['main']['temp']
    else:
        return None

import random

def Current_random():
    r = random.random()
    if r < 0.95:
        return round(random.uniform(0.28, 0.3), 2)
    elif r < 0.98:
        return round(random.uniform(0.3, 0.33), 2)
    else:
        return round(random.uniform(0.35, 0.4), 2)

import random

def calculate_vibration(current: float) -> float:
    """
    Converts current to a vibration value in the range [0.0, 1.0].

    Parameters:
        current (float): The input current value.

    Returns:
        float: Simulated vibration value.
    """
    # Normalize current to [0, 1] based on max expected current of 0.4
    normalized_current = current / 0.4

    # Add noise in the range [-0.05, 0.05]
    noise = random.uniform(-0.05, 0.05)

    # Compute vibration and clamp to [0.0, 0.4]
    vibration = round(normalized_current + noise, 2)
    return max(0.0, min(vibration, 1.0))



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


        ########## Current ########
        current = Current_random()


        ########## humidity ########
        # Try getting live humidity
        live_humidity = get_humidity()
        if live_humidity is not None:
            humidity = live_humidity  # Use API value only if available

        # Simulated fallback temperature
        temperature = round(random.uniform(30.0, 45.0), 2)
        # Try getting live temperature
        live_temp = get_temperature()
        if live_temp is not None:
            temperature = live_temp

         ########### Vibration ############
        live_vibration = calculate_vibration(current)
        if live_vibration is not None:
            vibration = live_vibration
       
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
        # logging.info("🔄 Received request to /predict/iotLive")

        # Ensure latest_prediction_result is available
        if latest_prediction_result is None:
            logging.warning("⚠️ No prediction result available. Returning 204.")
            return Response(status_code=204)

        #logging.info(f"✅ Returning prediction result: {latest_prediction_result}")
        return latest_prediction_result  # ✅ return flat structure

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

    with iot_buffer_lock:  # Thread-safe access
        try:
            now = datetime.now()

            # Clear buffer if data is too old
            if latest_iot_time and (now - latest_iot_time).total_seconds() > IOT_DATA_EXPIRY_SECONDS:
                latest_iot_data.clear()
                logging.info("🧹 Cleared stale buffer due to expiry")

            # Merge new data
            incoming_data = {k: v for k, v in data.dict(exclude_unset=True).items()}
            latest_iot_data.update(incoming_data)
            latest_iot_time = now

            # Log received and buffered fields
            logging.info(f"📥 Received fields this request: {list(incoming_data.keys())}")
            logging.info(f"📦 Current buffer state: {list(latest_iot_data.keys())}")

            # Calculate power if we have current and voltage
            if 'current' in latest_iot_data and 'voltage' in latest_iot_data:
                latest_iot_data['power'] = latest_iot_data['current'] * latest_iot_data['voltage']
                # logging.info(f"⚡ Calculated power: {latest_iot_data['power']}W")

                # --- Calculate vibration once current is available ---
                live_vibration = calculate_vibration(latest_iot_data['current'])
                if live_vibration is not None:
                    latest_iot_data['vibration'] = live_vibration
                    # logging.info(f"🟢 Calculated vibration: {live_vibration}")

                # --- Calculate humidity (if you have a function) ---
                live_humidity = get_humidity()
                if live_humidity is not None:
                       latest_iot_data['humidity'] = live_humidity
                       # logging.info(f"🟢 Calculated humidity: {live_humidity}")

            # Check required fields (excluding power since we calculate it)
            required_fields = {"voltage", "current", "temperature", "vibration", "humidity"}
            missing = required_fields - latest_iot_data.keys()

            if missing:
                logging.info(f"⏳ Awaiting fields: {', '.join(missing)}")
                return {
                    "status": "partial",
                    "received": list(latest_iot_data.keys()),
                    "missing": list(missing)
                }

            # Prepare data for prediction
            input_df = pd.DataFrame([latest_iot_data]).rename(columns={
                "voltage": "Voltage",
                "current": "Current",
                "temperature": "Temperature",
                "power": "Power",
                "vibration": "Vibration",
                "humidity": "Humidity"
            })[expected_fields]

            # Log input_df fields and values
            field_values = ', '.join(f"{col}={input_df.iloc[0][col]}" for col in input_df.columns)
            logging.info(f"📥 Final input DataFrame for prediction: {field_values}")

            # Make prediction
            probability = model.predict_proba(input_df)[0][1]
            prediction = int(probability > 0.5)

            # Update global state
            latest_prediction_result = {
                "prediction": prediction,
                "probability": float(probability),
                "timestamp": now.isoformat()
            }

            # Clear buffer for new data cycle
            latest_iot_data.clear()
            logging.info("✅ Prediction successful - buffer reset")

            return latest_prediction_result

        except Exception as e:
            logging.error("❌ Prediction failed:", exc_info=True)
            latest_iot_data.clear()
            return {
                "error": str(e),
                "prediction": -1,
                "probability": 0.0
            }
