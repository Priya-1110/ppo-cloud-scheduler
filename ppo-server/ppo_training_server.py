"""
PPO Inference Socket Server
---------------------------
Purpose:
    Exposes a minimal TCP socket server that loads a pretrained PPO model
    (Stable-Baselines3) and serves inference for a single request at a time.

Protocol (very simple):
    Client -> Server: JSON string with key "state", where payload['state'] is itself
                      a *stringified* JSON list of floats (length = STATE_DIM).
                      Example:
                      {"state": "[0.62, 0.30, 0.12, 0.0, 0.45]"}

    Server -> Client: stringified integer action (e.g., "0", "1", or "2"), representing
                      the chosen cloud index.

Notes:
    - The double-JSON for "state" is intentional to match the existing Java sender.
      We parse JSON once to get 'payload', then json.loads(payload['state']) again
      to convert the inner stringified list to a Python list.
    - This server is single-threaded and handles one client connection at a time.
      For concurrent requests, wrap the client handler in a thread or process pool.
    - Prints include emojis for quick visual tracing during demos/logs.
"""

import socket
import json
import numpy as np
from stable_baselines3 import PPO

# Server & model config
HOST = 'localhost'
PORT = 5055
STATE_DIM = 5  # Expected length of input state vector
MODEL_PATH = "ppo_v2.zip"

# Load trained PPO model
model = PPO.load(MODEL_PATH)
print(f"✅ PPO Model loaded from {MODEL_PATH}")

# Create and start socket server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"✅ PPO Server running on {HOST}:{PORT}...")

while True:
    client, addr = server.accept()
    data = client.recv(4096).decode()

    try:
        payload = json.loads(data)  # Parse incoming JSON payload
        state = np.array(json.loads(payload['state']), dtype=np.float32)  # Parse stringified state list

        if len(state) != STATE_DIM:
            raise ValueError(f"❌ Expected {STATE_DIM}-length state vector, got {len(state)}")

        action, _ = model.predict(state)  # Predict best action using PPO model
        response = str(int(action))  # Convert action to string for sending

        print(f"🧠 Predicted Cloud Index: {response}")  # Log prediction
        client.send(response.encode())  # Send prediction back to client

    except Exception as e:
        print(f"❌ Error: {e}\n⚠️ Payload: {data}")
    finally:
        client.close()  # Close client connection
