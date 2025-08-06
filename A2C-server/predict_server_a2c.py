import socket
import json
import numpy as np
from stable_baselines3 import A2C

# Load trained A2C model
model = A2C.load("a2c_from_ppo_model_v2", custom_objects={"lr_schedule": lambda _: 0.0003})
print("✅ A2C model loaded successfully")

HOST = 'localhost'
PORT = 9999

# Run server loop
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
                    request = json.loads(data)
                    state = np.array(request['state']).reshape(1, -1)
                    print(f"📥 Received state: {state}")

                    action, _ = model.predict(state, deterministic=True)
                    print(f"📤 Predicted action: {action[0]}")

                    response = {'cloud': int(action[0])}
                    conn.sendall(json.dumps(response).encode())

                except Exception as e:
                    error_msg = f"⚠️ Error processing request: {str(e)}"
                    print(error_msg)
                    conn.sendall(json.dumps({'error': error_msg}).encode())
    except KeyboardInterrupt:
        print("❌ Server manually stopped.")
