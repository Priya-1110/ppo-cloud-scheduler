import socket
import json
import numpy as np
from joblib import load

# Load trained model
model = load("xgboost_scheduler.pkl")
print("✅ XGBoost model loaded.")

# Server config
HOST = "localhost"
PORT = 65432

# Create socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print(f"🚀 Listening on {HOST}:{PORT}...")

while True:
    conn, addr = server.accept()
    print(f"📥 Connected by {addr}")
    data = conn.recv(1024).decode()

    try:
        task_features = json.loads(data)  # Expecting a JSON list
        print(f"🧠 Input: {task_features}")

        # Convert to 2D array for prediction
        features_array = np.array(task_features).reshape(1, -1)
        prediction = model.predict(features_array)[0]

        conn.send(str(prediction).encode())
        print(f"✅ Predicted Cloud: {prediction}")

    except Exception as e:
        print("❌ Error:", e)
        conn.send(b"error")

    conn.close()
