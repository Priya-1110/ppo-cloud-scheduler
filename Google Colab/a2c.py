# ============================================================
# Title: A2C Training on PPO-Style Offline Dataset (Colab)
# Purpose:
#   - Load a preprocessed 5D state dataset (StateVec/Action)
#   - Define a simple offline Gymnasium env over CSV rows
#   - Train a Stable-Baselines3 A2C model with exploration
# Inputs:
#   - CSV file: 'ppo_training_dataset_cleaned_5f.csv' (or similar)
#   - Columns required: 'StateVec' (list of 5 floats), 'Action' (int in {0,1,2})
# Output:
#   - Saved model: 'a2c_from_ppo_model_v2'
# Notes:
#   - Uses per-row supervised-style reward shaping (match label = +1, else small penalty)
#   - Normalizes states per-feature using max over the dataset (0..1)
#   - Action space is 3 (e.g., clouds: 0=AWS, 1=Azure, 2=GCP)
# ============================================================

# ✅ Step 1: Install required libraries
# !pip install gymnasium==0.29.1 stable-baselines3[extra]

# ✅ Step 2: Imports
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env

# ✅ Step 3: Load PPO dataset
from google.colab import files

df = pd.read_csv("ppo_training_dataset_cleaned_5f.csv")  # Replace if needed

# Basic schema checks for required columns
if 'StateVec' not in df.columns or 'Action' not in df.columns:
    raise ValueError("ERROR: Dataset must contain 'StateVec' and 'Action' columns.")

# Parse StateVec lists and ensure Action integers
df['StateVec'] = df['StateVec'].apply(eval)
df['Action'] = df['Action'].astype(int)
df = df.dropna(subset=['StateVec', 'Action'])

# ✅ Step 4: Custom offline environment
class CloudSchedulerCSVEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Cache states and labels in memory for fast iteration
        self.states = [np.array(s, dtype=np.float32) for s in df['StateVec']]
        self.labels = df['Action'].tolist()
        self.n_tasks = len(self.states)
        # Per-feature max for simple 0..1 normalization
        self.max_vals = np.max(np.vstack(self.states), axis=0)
        # Observation: 5D normalized vector; Action: 3 discrete choices
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.current_step = 0

    def _normalize_state(self, state):
        return (state / self.max_vals).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        return self._normalize_state(self.states[self.current_step]), {}

    def step(self, action):
        correct_action = self.labels[self.current_step]

        # ✅ Reward shaping: +1 if correct label; small graded penalty otherwise
        if action == correct_action:
            reward = 1.0
        else:
            reward = -0.25 * abs(action - correct_action)  # Gradual penalty

        # Advance to next sample
        self.current_step += 1
        done = self.current_step >= self.n_tasks

        # Next observation or zeros if episode ended
        if not done:
            obs = self._normalize_state(self.states[self.current_step])
        else:
            obs = np.zeros(5, dtype=np.float32)

        return obs, reward, done, False, {}

# ✅ Step 5: Environment and model setup
env = CloudSchedulerCSVEnv()
check_env(env)  # Sanity-check spaces and API

# ✅ Step 6: Train with stronger exploration
model = A2C(
    "MlpPolicy",
    env,
    verbose=1,
    ent_coef=0.05,         # 🔥 Boost exploration
    learning_rate=0.0007,
    n_steps=5,
    gamma=0.99
)

model.learn(total_timesteps=50000)  # 🔁 Train longer
model.save("a2c_from_ppo_model_v2")

print("✅ A2C model trained with enhanced reward + exploration!")
