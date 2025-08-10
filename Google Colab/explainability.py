# ============================================================
# Title: Rule-Based Explainability Table for Scheduler Logs
# Purpose:
#   - Read PPO, A2C, and DQN simulation logs
#   - Apply simple, human-readable reasoning rules
#   - Export an explainability table (per task, per model)
# Input:  ppo_log.csv, A2C_log.csv, dqn_log.csv
# Output: explainability_table.csv
# ============================================================

import pandas as pd

# Load PPO, A2C, DQN logs
models = {
    "PPO": pd.read_csv("ppo_log.csv"),
    "A2C": pd.read_csv("A2C_log.csv"),
    "DQN": pd.read_csv("dqn_log.csv")
}

rows = []

# Iterate each model’s rows and derive a text reason
for model_name, df in models.items():
    for idx, row in df.iterrows():
        sla = row['SLAMet']
        cost = row['CPUCost']
        sla_dur = row['SLADuration']
        exec_time = row['ExecutionTime']
        cloud = row['SelectedCloud']
        task_id = row['TaskID']

        # Rule-based reasoning (ordered by priority)
        if sla_dur < 0.1 and cost > 150:
            reason = "Tight deadline forced a fast cloud despite cost"
        elif sla_dur < 0.1:
            reason = "Low SLA Deadline led to faster cloud selection"
        elif cost > 150:
            reason = "High cost led to choosing cheaper cloud"
        elif cost < 100 and sla_dur > 0.1:
            reason = "Long SLA allowed cost-optimized scheduling"
        elif exec_time > sla_dur:
            reason = "Execution time exceeded SLA deadline"
        else:
            reason = "Balanced decision between SLA and cost"

        rows.append({
            "Model": model_name,
            "TaskID": task_id,
            "SelectedCloud": cloud,
            "SLA": sla,
            "Reason": reason
        })

# Save updated explainability table
explain_df = pd.DataFrame(rows)
explain_df.to_csv("explainability_table.csv", index=False)
print("✅ New explainability table saved as 'explainability_table.csv'")