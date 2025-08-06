import socket
import json
import numpy as np
from stable_baselines3 import PPO

# === CONFIG ===
HOST = 'localhost'
PORT = 5055
STATE_DIM = 5  # CPU, MEM, Start Time, SLA, Cost
MODEL_PATH = "ppo_v3.zip"

# === Load Trained PPO Model ===
model = PPO.load(MODEL_PATH)
print(f"✅ PPO Model loaded from {MODEL_PATH}")

# === Start Socket Server ===
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print(f"✅ PPO Inference Server running on {HOST}:{PORT}...")

while True:
    client, addr = server.accept()
    data = client.recv(4096).decode()

    try:
        payload = json.loads(data)
        # Fix: Parse stringified state list
        state = np.array(json.loads(payload['state']), dtype=np.float32)

        if len(state) != STATE_DIM:
            raise ValueError(f"❌ Expected {STATE_DIM}-length state vector, got {len(state)}")

        # Predict using PPO model
        action, _ = model.predict(state)
        response = str(int(action))

        # Send response
        client.send(response.encode())

    except Exception as e:
        print(f"❌ Error: {e}\n⚠️ Payload: {data}")
    finally:
        client.close()
