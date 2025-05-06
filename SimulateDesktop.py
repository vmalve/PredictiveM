import requests
import time
import random
import logging
import traceback
import argparse

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)

def generate_sensor_data():
    voltage = round(random.uniform(180.0, 240.0), 2)
    current = round(random.uniform(0.5, 10.0), 2)
    temperature = round(random.uniform(20.0, 100.0), 2)
    power = round(voltage * current, 2)
    vibration = round(random.uniform(0.0, 1.0), 2)

    data = {
        "voltage": voltage,
        "current": current,
        "temperature": temperature,
        "power": power,
        "vibration": vibration
    }
    logging.debug(f"Generated sensor data: {data}")
    return data

def send_data(api_url, interval, count):
    tries = 0
    while True:
        data = generate_sensor_data()
        try:
            logging.info(f"Sending data to {api_url}")
            response = requests.post(api_url, json=data, timeout=10)
            response.raise_for_status()
            logging.info(f"✅ Response: {response.json()}")

        except requests.exceptions.RequestException:
            logging.error("❌ Request error:")
            logging.error(traceback.format_exc())

        except Exception:
            logging.error("❌ Unexpected error:")
            logging.error(traceback.format_exc())

        tries += 1
        if count > 0 and tries >= count:
            break
        time.sleep(interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT Simulator")

    parser.add_argument("--url", type=str, default="https://predictivem.onrender.com/update_iot_data/",
                        help="URL of the prediction API (default: localhost)")
    parser.add_argument("--once", action="store_true", help="Send data only once")
    parser.add_argument("--interval", type=int, default=5, help="Time between sends (seconds)")
    parser.add_argument("--count", type=int, default=0, help="Number of times to send (0 = infinite)")

    args = parser.parse_args()

    final_count = 1 if args.once else args.count

    logging.info("🚀 IoT Simulator Started")
    send_data(args.url, args.interval, final_count)
