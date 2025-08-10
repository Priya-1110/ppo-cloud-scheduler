# 🚀 Real-Time, Cost-Aware Resource Scheduling in Multi-Cloud Systems Using PPO

## 📌 Overview

This project implements an **AI-driven cloud resource scheduler** that intelligently allocates tasks across multi-cloud providers (AWS, Azure, GCP) using **Proximal Policy Optimization (PPO)**.
The system is simulated in **iFogSim2 (Java)** with real-time decision-making via a **Python socket server** running a pretrained PPO model.
It compares PPO against **Round Robin (RR)**, **First Come First Serve (FCFS)**, **Deep Q-Network (DQN)**, and **Advantage Actor-Critic (A2C)**.

The goal is to **maximize SLA compliance**, **minimize CPU cost**, and **reduce execution time**.

---

## 🎯 Objectives

- Develop an **intelligent task scheduler** for multi-cloud environments.
- Integrate **Java simulation** with **Python-based DRL models** via sockets.
- Evaluate performance vs traditional and other DRL schedulers.
- Provide a **Streamlit dashboard** for visualization.

---

## 🛠 Architecture

**Key Components:**

- **iFogSim2 (Java)** – Simulates the multi-cloud infrastructure & workload.
- **PPOClient.java** – Sends task state to Python server, receives scheduling decision.
- **PPO Python Server** – Runs PPO model (Stable-Baselines3), returns optimal cloud selection.
- **Evaluation Logs** – SLA, cost, and execution metrics for each task.
- **Streamlit Dashboard** – Visualizes and compares scheduler performance.

```
[ iFogSim2 Simulation ] ⇄ [ PPOClient.java ] ⇄ [ Python PPO Server ]
                                 ↓
                         [ Evaluation Logs ]
                                 ↓
                      [ Streamlit Dashboard ]
```

---

## 📂 Project Structure

```
├── A2C-server/
│   ├── a2c_from_ppo_model.py       # A2C scheduler logic from PPO model
│   ├── predict_server.py           # Socket server for A2C inference
│
├── DQN-server/
│   ├── dqn_predict_server.py       # Socket server for DQN inference
│   ├── dqn_v1.zip                  # Pretrained DQN model
│
├── Google Colab/
│   ├── Datasets/                   # Offline training datasets
│   ├── a2c.py                       # A2C training script
│   ├── dqn.py                       # DQN training script
│   ├── explainability.py            # SHAP/LIME explainability code
│   ├── ppo.py                       # PPO training script
│
├── java-iFogSim/
│   ├── A2CClient.java               # Java socket client for A2C
│   ├── DQNClient.java               # Java socket client for DQN
│   ├── FCFScheduler.java            # FCFS scheduling logic
│   ├── MultiCloudSchedulingSim.java # Main simulation file
│   ├── PPOClient.java               # Java socket client for PPO
│   ├── RoundRobinScheduler.java     # Round Robin scheduling logic
│
├── ppo-server/
│   ├── ppo_training_server.py       # PPO socket server
│   ├── ppo_v2.zip                   # Pretrained PPO model
│
├── results/                         # Evaluation logs (CSV format)
│   ├── A2C_log.csv
│   ├── dqn_log.csv
│   ├── explainability_table.csv
│   ├── fcfs_log.csv
│   ├── ppo_log.csv
│   ├── round_robin_log.csv
│
├── venv/                            # Python virtual environment
│
├── app.py                           # Streamlit dashboard
├── README.md
├── requirements.txt                 # Python dependencies
```

---

## ⚙️ Installation & Setup

### **1️⃣ Java (iFogSim2) Setup**

1. Install **Eclipse IDE**.
2. Clone iFogSim2:

   ```bash
   git clone https://github.com/Cloudslab/iFogSim2.git
   ```

3. Add project files (`MultiCloudSchedulingSim.java`, `PPOClient.java`) to `org.fog.test` package.

---

### **2️⃣ Python Environment Setup**

```bash
# Create and activate virtual env
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# Install dependencies
pip install stable-baselines3==1.8.0 gym==0.21.0 pandas streamlit plotly
```

---

### **3️⃣ PPO Server**

Run the PPO server before starting the Java simulation:

```bash
python python/ppo_training_server.py
```

---

### **4️⃣ Running the Simulation**

In Eclipse:

- Open `MultiCloudSchedulingSim.java`.
- Run the simulation.
- Watch **console output** for real-time cloud selections and SLA logs.

---

### **5️⃣ Streamlit Dashboard**

```bash
cd dashboard
streamlit run app.py
```

Then open the link in your browser.

---

## 📊 Results

| Scheduler | SLA (%) | CPU Cost | Time (ms) |
| --------- | ------- | -------- | --------- |
| **PPO**   | 78.47   | 126.9    | 0.65      |
| RR        | 33.34   | 127.0    | 1.25      |
| FCFS      | 32.5    | 128.2    | 1.26      |
| A2C       | 51.0    | 127.4    | 0.93      |
| DQN       | 49.6    | 128.1    | 0.91      |

---

## 📌 Novel Contributions

- **Hybrid offline + real-time PPO training** for better adaptability.
- **State vector with SLA deadlines, CPU cost, and RAM** for richer decision-making.
- **Explainability-ready** design for SHAP/LIME (future extension).
- **Unified dashboard** for scheduler comparison.

---

## 📚 References

- iFogSim2: [https://github.com/Cloudslab/iFogSim2](https://github.com/Cloudslab/iFogSim2)
- Stable-Baselines3: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)

---
