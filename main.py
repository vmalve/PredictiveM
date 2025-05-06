from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime, timedelta

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
model = joblib.load("random_forest_model.pkl")

# Sensor data structure
class SensorData(BaseModel):
    voltage: float
    current: float
    temperature: float
    power: float
    vibration: float

# Store latest IoT data and timestamp
latest_iot_data = None
latest_iot_time = None

@app.get("/")
def read_root():
    return {"status": "IoT ML API Running"}

# Manual prediction (no effect on IoT state)
@app.post("/predict/")
def predict(data: SensorData):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]
    return {"prediction": int(pred), "probability": float(proba)}

# Update IoT data (e.g., from backend device or simulator)
@app.post("/update_iot_data")
def update_iot_data(data: SensorData):
    global latest_iot_data, latest_iot_time
    latest_iot_data = data.dict()
    latest_iot_time = datetime.now()
    return {"status": "IOT data received"}

# Get prediction from latest IoT data
@app.get("/get_prediction")
def get_prediction():
    global latest_iot_data, latest_iot_time

    if latest_iot_data is None or latest_iot_time is None:
        return {"prediction": "No data", "probability": 0.0}

    if datetime.now() - latest_iot_time > timedelta(minutes=IOT_DATA_EXPIRY_MINUTES):
        return {"prediction": "No recent data", "probability": 0.0}

    df = pd.DataFrame([latest_iot_data])
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]
    return {"prediction": int(pred), "probability": float(proba)}
