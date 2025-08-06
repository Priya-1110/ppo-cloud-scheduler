import pandas as pd

# Load each model log
ppo = pd.read_csv("ppo_log.csv")
rr = pd.read_csv("round_robin_log.csv")
fcfs = pd.read_csv("fcfs_log.csv")
a2c = pd.read_csv("A2C_log.csv")
dqn = pd.read_csv("dqn_log.csv")  # ✅ NEW

# Define evaluation helper
def evaluate_model(df, model_name):
    total = len(df)
    sla_met = df[df['SLAMet'].astype(str).str.upper() == 'YES']
    sla_percent = (len(sla_met) / total) * 100

    avg_cpu_cost = df['CPUCost'].mean()

    if 'ExecutionTime' in df.columns:
        avg_exec_time = df['ExecutionTime'].mean()
    else:
        avg_exec_time = None

    return {
        "Model": model_name,
        "SLA %": round(sla_percent, 2),
        "Avg CPUCost": round(avg_cpu_cost, 4),
        "Avg Execution Time": round(avg_exec_time, 4) if avg_exec_time else "N/A"
    }

# Evaluate all models
results = [
    evaluate_model(ppo, "PPO"),
    evaluate_model(rr, "Round Robin"),
    evaluate_model(fcfs, "FCFS"),
    evaluate_model(a2c, "A2C"),
    evaluate_model(dqn, "DQN")  # ✅ NEW
    
]

# Display comparison
df_results = pd.DataFrame(results)
print("📊 All Model Results:")
print(df_results)

# Sort by best SLA and lowest cost
df_sorted = df_results.sort_values(by=["SLA %", "Avg CPUCost"], ascending=[False, True])
print("\n🏆 Best Model Summary:")
print(df_sorted.head(1))
