"""
A2C Inference Socket Server
---------------------------
Purpose:
    Runs a lightweight TCP socket server that loads a pretrained A2C model
    (Stable-Baselines3) and returns a discrete action (cloud index) for a
    given input state. Designed for simple, synchronous inference from an
    external simulator (e.g., iFogSim/Java).

Protocol:
    Client -> Server:
        JSON string with key "state" containing a direct JSON list of floats.
        (No double-encoding.)
        Example:
            {"state": [0.72, 0.33, 0.15, 1.0, 0.28]}

    Server -> Client:
        JSON string with key "cloud" (int), representing the chosen cloud index.
        Example:
            {"cloud": 2}

Key Notes:
    - Input shape must match the training environment’s observation dimension.
      This server reshapes to (1, -1) for SB3’s predict() API.
    - Inference is deterministic (no exploration) for reproducible results.
    - Single-threaded: one connection at a time. For concurrency, wrap
      connection handling in threads or async I/O.
    - The lr_schedule override avoids SB3 load-time compatibility issues.
    - Stop gracefully with Ctrl+C (KeyboardInterrupt).

Version/Compatibility:
    - Requires stable-baselines3 and its dependencies (PyTorch).
    - Model file name is "a2c_from_ppo_model_v2" (adjust if different).
"""

import socket
import json
import numpy as np
from stable_baselines3 import A2C

# Load trained A2C model (override lr_schedule for compatibility)
model = A2C.load("a2c_from_ppo_model_v2", custom_objects={"lr_schedule": lambda _: 0.0003})
print("✅ A2C model loaded successfully")

HOST = 'localhost'
PORT = 9999  # Change if another server (e.g., DQN) also uses 9999

# Run server loop (auto-closes on exit)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))
    server_socket.listen()
    print(f"✅ A2C Socket Server listening on {HOST}:{PORT}...")

    try:
        while True:
            conn, addr = server_socket.accept()
            with conn:
                print(f"🔌 Connected by {addr}")
                data = conn.recv(1024).decode()
                if not data:
                    continue

                try:
                    request = json.loads(data)                               # Parse JSON
                    state = np.array(request['state']).reshape(1, -1)       # 2D for SB3
                    print(f"📥 Received state: {state}")

                    action, _ = model.predict(state, deterministic=True)     # Inference only
                    print(f"📤 Predicted action: {action[0]}")

                    response = {'cloud': int(action[0])}                     # Match your protocol
                    conn.sendall(json.dumps(response).encode())

                except Exception as e:
                    error_msg = f"⚠️ Error processing request: {str(e)}"
                    print(error_msg)
                    conn.sendall(json.dumps({'error': error_msg}).encode())

    except KeyboardInterrupt:
        print("❌ Server manually stopped.")
