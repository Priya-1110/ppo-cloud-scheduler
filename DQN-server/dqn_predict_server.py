"""
DQN Inference Socket Server
---------------------------
Purpose:
    Hosts a simple TCP socket server that loads a pretrained DQN model 
    (Stable-Baselines3) and provides inference for a single state request at a time.

Protocol:
    Client -> Server:
        JSON string containing key "state", where "state" is a direct JSON list of 
        floats (no double-encoding like in PPO). 
        Example:
            {"state": [0.75, 0.45, 0.20, 1.0, 0.35]}

    Server -> Client:
        JSON string containing key "action", where "action" is the integer index 
        of the chosen cloud.
        Example:
            {"action": 1}

Key Notes:
    - The input state must be a list with the same dimensionality used during DQN training.
      Here, it is reshaped to (1, -1) for compatibility with SB3 predict().
    - This server is synchronous and handles one connection at a time.
      For concurrent handling, wrap the client connection in threads or async I/O.
    - The learning rate schedule is overridden on load with a fixed lambda to 
      avoid SB3 incompatibility warnings.
    - Server stops gracefully with a KeyboardInterrupt (Ctrl+C).
"""

import socket
import json
import numpy as np
from stable_baselines3 import DQN

# ✅ Load trained DQN model (keeps saved lr_schedule compatible)
model = DQN.load("dqn_v1", custom_objects={"lr_schedule": lambda _: 0.0003})
print("✅ DQN model loaded successfully")

HOST = 'localhost'
PORT = 9999

# ✅ Run socket server (auto-closes on exit)
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.bind((HOST, PORT))          # Bind to host:port
    server_socket.listen()                    # Start listening for clients
    print(f"✅ DQN Socket Server listening on {HOST}:{PORT}...")

    try:
        while True:
            conn, addr = server_socket.accept()   # Block until a client connects
            with conn:
                print(f"🔌 Connected by {addr}")
                data = conn.recv(1024).decode()   # Receive request bytes → str
                if not data:
                    continue

                try:
                    request = json.loads(data)                    # Parse JSON
                    state = np.array(request['state']).reshape(1, -1)  # 2D for SB3
                    print(f"📥 Received state: {state}")

                    action, _ = model.predict(state, deterministic=True)  # Inference (no exploration)
                    print(f"📤 Predicted action: {action[0]}")

                    # ✅ Send back JSON response with the selected action
                    response = {'action': int(action[0])}
                    conn.sendall(json.dumps(response).encode())

                except Exception as e:
                    # Return error as JSON (useful for debugging client-side)
                    error_msg = f"⚠️ Error processing request: {str(e)}"
                    print(error_msg)
                    conn.sendall(json.dumps({'error': error_msg}).encode())

    except KeyboardInterrupt:
        print("❌ Server manually stopped.")
