# ============================================================
# Title: Build PPO-Style Training Dataset (Merge FCFS/RR/Custom)
# Purpose: Convert FCFS/RR logs + a custom dataset into a unified
#          PPO-style CSV with 5D StateVec, Action, Reward, SLAMet, CPUCost.
# Output:  ppo_training_dataset_final.csv (auto-download in Colab)
# ============================================================

# ⬇️ STEP 1: Install if needed (Colab)
# !pip install pandas --quiet

import pandas as pd
from google.colab import files

# STEP 2: Load uploaded files
df_fcfs = pd.read_csv("fcfs_log.csv")                 # baseline: FCFS run logs
df_rr   = pd.read_csv("round_robin_log.csv")          # baseline: Round-Robin run logs
df_custom = pd.read_csv("cloud_task_scheduling_dataset.csv")  # custom source dataset

# ---------- Helpers ----------
def minmax(x, eps=1e-9):
    """Safe min-max normalization in [0,1] with epsilon to avoid divide-by-zero."""
    return (x - x.min()) / (x.max() - x.min() + eps)

# ✅ STEP 3: Format FCFS/RR logs → 5D StateVec aligned with PPO server order:
# [CPU, MEM, StartTime, SLA, Cost]
def process_log(df, label):
    df = df.copy()
    # Normalize what we have; CPU/MEM are unknown here → pad as 0.0
    start_norm = minmax(df["StartTime"])
    sla_norm   = minmax(df["SLADuration"])
    cost_norm  = minmax(df["CPUCost"])
    df["StateVec"] = [
        [0.0, 0.0, float(s), float(a), float(c)]
        for s, a, c in zip(start_norm, sla_norm, cost_norm)
    ]
    df["Action"] = df["SelectedCloud"].astype(int)
    df["Reward"] = df["SLAMet"].astype(str).str.upper().eq("YES").astype(int)
    df["Source"] = label
    return df[["StateVec", "Action", "Reward", "SLAMet", "CPUCost", "Source"]]

ppo_fcfs = process_log(df_fcfs, "FCFS")
ppo_rr   = process_log(df_rr, "RoundRobin")

# ✅ STEP 4: Process your custom dataset → map to PPO 5D order
# Map: CPU%→CPU, RAM(MB)→MEM, Start/SLA unknown→0/1, Cost≈Execution_Time
cpu_norm = df_custom["CPU_Usage (%)"] / 100.0
mem_norm = minmax(df_custom["RAM_Usage (MB)"])
cost_norm = minmax(df_custom["Execution_Time (s)"])
df_custom["StateVec"] = [
    [float(c), float(m), 0.0, 1.0, float(cost)]  # assume SLA present (1.0)
    for c, m, cost in zip(cpu_norm, mem_norm, cost_norm)
]
df_custom["Action"] = df_custom["Target (Optimal Scheduling)"].astype(int)
df_custom["Reward"] = 1  # assume SLA met for originals
df_custom["SLAMet"] = "YES"
df_custom["CPUCost"] = df_custom["Execution_Time (s)"]
df_custom["Source"] = "Original"
ppo_cloud = df_custom[["StateVec", "Action", "Reward", "SLAMet", "CPUCost", "Source"]]

# ✅ STEP 5: Combine and (optionally) sample ~4000 rows
combined = pd.concat([ppo_fcfs, ppo_rr, ppo_cloud], ignore_index=True)

n = min(4000, len(combined))  # cap at 4000 if available
final_dataset = combined.sample(n=n, random_state=42).reset_index(drop=True)

# ✅ STEP 6: Save & download (Colab)
final_dataset.to_csv("ppo_training_dataset_final.csv", index=False)
print("✅ Final PPO dataset shape:", final_dataset.shape)
print(final_dataset.Source.value_counts())

files.download("ppo_training_dataset_final.csv")
