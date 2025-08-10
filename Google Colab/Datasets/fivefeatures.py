# ============================================================
# Title: PPO Training Dataset Cleaner (5-Feature Format)
# Purpose: Ensure each row has a 5D StateVec, add NextState, normalize fields,
#          and save a cleaned CSV ready for PPO training/inference demos.
# Usage (Colab): Upload `ppo_training_dataset_final.csv`, run this cell to get
#                `ppo_training_dataset_cleaned_5f.csv` and auto-download it.
# ============================================================

import pandas as pd
import numpy as np
from ast import literal_eval

# Step 1: Upload original CSV (Colab helper import)
from google.colab import files

# Step 2: Load and inspect
df = pd.read_csv("ppo_training_dataset_final.csv")
df.columns = df.columns.str.strip()
print("✅ Columns:", df.columns)

# Step 3: Parse StateVec from string → list
df['StateVec'] = df['StateVec'].apply(literal_eval)

# Step 4: Pad/trim StateVec to exactly 5 features
def fix_vec(v):
    v = list(v)
    return v[:5] if len(v) > 5 else v + [0.0]*(5 - len(v))

df['StateVec'] = df['StateVec'].apply(fix_vec)

# Step 5: Create NextState (same as StateVec for now)
df['NextState'] = df['StateVec']

# Step 6: Clean up reward, done (if present / missing)
if 'Reward' in df.columns:
    df['Reward'] = pd.to_numeric(df['Reward'], errors='coerce').fillna(0)

if 'Done' not in df.columns:
    df['Done'] = False

# Step 7: Save cleaned CSV
df.to_csv("ppo_training_dataset_cleaned_5f.csv", index=False)
print("✅ Saved as ppo_training_dataset_cleaned_5f.csv")

# Step 8: Download (Colab)
files.download("ppo_training_dataset_cleaned_5f.csv")
