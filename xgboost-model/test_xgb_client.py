import socket
import json

HOST = "localhost"
PORT = 65432

# 🧠 15 Sample Inputs: [CPU_Usage, RAM, Disk_IO, Net_IO, Priority, Cost, SLA_Level]
test_inputs = [
    [55.0, 4096.0, 100.0, 25.0, 2.5, 64.0, 2],     # Medium load
    [90.0, 16384.0, 300.0, 80.0, 0.8, 200.0, 0],   # Heavy load, high cost
    [20.0, 2048.0, 50.0, 10.0, 5.0, 10.0, 3],      # Low load, low cost
    [65.0, 6144.0, 110.0, 35.0, 1.5, 100.0, 1],    # Moderate
    [10.0, 1024.0, 20.0, 5.0, 10.0, 5.0, 3],       # Very light
    [95.0, 32768.0, 500.0, 120.0, 0.5, 300.0, 0],  # Extreme heavy
    [35.0, 4096.0, 70.0, 18.0, 3.0, 40.0, 2],      # Light-medium
    [70.0, 8192.0, 150.0, 40.0, 1.2, 128.0, 1],    # Heavy-medium
    [50.0, 6144.0, 90.0, 30.0, 2.0, 85.0, 2],      # Average task
    [40.0, 3072.0, 80.0, 22.0, 3.5, 60.0, 2],      # Moderate-light
    [75.0, 12288.0, 200.0, 60.0, 1.0, 150.0, 1],   # Strong compute, high SLA
    [25.0, 2048.0, 45.0, 12.0, 6.0, 20.0, 3],      # Low power edge case
    [85.0, 16384.0, 250.0, 70.0, 0.9, 180.0, 0],   # Heavy workload
    [60.0, 8192.0, 130.0, 38.0, 1.8, 110.0, 1],    # Balanced
    [30.0, 2048.0, 60.0, 14.0, 4.5, 25.0, 2],      # Low-mid with urgency
]

# 🔁 Loop through and test each input
for idx, sample_input in enumerate(test_inputs):
    data = json.dumps(sample_input)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(data.encode())
        prediction = s.recv(1024).decode()
        print(f"{idx+1:02d}. Input: {sample_input} → 🧠 Predicted Cloud Index: {prediction}")
