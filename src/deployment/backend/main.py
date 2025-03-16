import asyncio
import json
import numpy as np
import tensorflow.lite as tflite
import websockets
from collections import deque

# Sensor server URL
SENSOR_SERVER_URL = "ws://192.168.81.38:8080/sensors/connect?types=[\"android.sensor.accelerometer\",\"android.sensor.gyroscope\"]"
# WebSocket server for frontend
FRONTEND_HOST = "0.0.0.0"
FRONTEND_PORT = 8000

# Activity labels
activity_labels = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
    7: "STAND_TO_SIT", 8: "SIT_TO_STAND", 9: "SIT_TO_LIE",
    10: "LIE_TO_SIT", 11: "STAND_TO_LIE", 12: "LIE_TO_STAND"
}

# Load TFLite models
def load_model(path):
    return tflite.Interpreter(model_path=path)

rnn_model = load_model("models/rnn50_model_quantized.tflite")
lstm_model = load_model("models/lstm50_model_quantized.tflite")
gru_model = load_model("models/gru50_model_quantized.tflite")

for model in [rnn_model, lstm_model, gru_model]:
    model.allocate_tensors()

# Run inference
def run_model(model, input_data):
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    
    model.set_tensor(input_details[0]['index'], input_data)
    model.invoke()
    
    output_data = model.get_tensor(output_details[0]['index'])
    return int(np.argmax(output_data))  # Get the predicted label index

# Store frontend clients
connected_clients = set()

# WebSocket server for frontend (display.html)
async def websocket_handler(websocket):
    connected_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(0.1)  # Prevents blocking
    finally:
        connected_clients.remove(websocket)

# Send predictions to frontend
async def send_predictions(predictions):
    if connected_clients:
        message = json.dumps(predictions)
        to_remove = set()
        for client in connected_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                to_remove.add(client)  # Mark client for removal

        # Remove disconnected clients
        for client in to_remove:
            connected_clients.remove(client)

data_buffer = deque(maxlen=50)
latest_prediction = {"RNN": None, "LSTM": None, "GRU": None}

sensor_data = {"acc": None, "gyro": None}  # Store latest sensor values
# WebSocket client for sensor data
async def sensor_client():
    async with websockets.connect(SENSOR_SERVER_URL) as ws:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            sensor_type = data.get("type", "")
            values = data.get("values", [])
            
            if len(values) == 3:
                if sensor_type == "android.sensor.accelerometer":
                    sensor_data["acc"] = values  # Store accelerometer data
                elif sensor_type == "android.sensor.gyroscope":
                    sensor_data["gyro"] = values  # Store gyroscope data

                # Process when both sensors have new data
                if sensor_data["acc"] is not None and sensor_data["gyro"] is not None:
                    sample = sensor_data["acc"] + sensor_data["gyro"]  # Combine acc + gyro
                    data_buffer.append(sample)

                    # Ensure 50 samples before predicting
                    if len(data_buffer) == 50:
                        input_data = np.array(data_buffer, dtype=np.float32).reshape(1, 50, 6)

                        latest_prediction["RNN"] = run_model(rnn_model, input_data)
                        latest_prediction["LSTM"] = run_model(lstm_model, input_data)
                        latest_prediction["GRU"] = run_model(gru_model, input_data)

                        print("✅ Prediction Updated:", latest_prediction)
            # Send predictions to frontend
            await send_predictions(latest_prediction)

# Start both WebSocket server (for frontend) and client (for sensor data)
async def main():
    sensor_task = asyncio.create_task(sensor_client())
    server = await websockets.serve(websocket_handler, FRONTEND_HOST, FRONTEND_PORT)
    print(f"✅ WebSocket Server Running at ws://{FRONTEND_HOST}:{FRONTEND_PORT}/ws")
    await asyncio.gather(sensor_task, server.wait_closed())

if __name__ == "__main__":
    asyncio.run(main())
