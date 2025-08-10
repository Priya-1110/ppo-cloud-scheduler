# ============================================================
# Title: DQN Training on PPO-Style Dataset 
# Purpose:
#   - Load a 5D StateVec dataset with labeled actions
#   - Define a simple offline Gymnasium env with reward shaping
#   - Train a Stable-Baselines3 DQN model and save it
# Inputs:
#   - CSV: 'ppo_training_dataset_cleaned_5f.csv'
#   - Columns: 'StateVec' (list of 5 floats), 'Action' (int in {0,1,2})
# Output:
#   - Model: 'dqn_v1.zip'
# ============================================================

# ✅ Step 1: Install dependencies
# pip install stable-baselines3[extra] gymnasium==0.29.1

# ✅ Step 2: Imports
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# ✅ Step 3: Load dataset
df = pd.read_csv("ppo_training_dataset_cleaned_5f.csv")

# Evaluate and convert state/next state
df['StateVec'] = df['StateVec'].apply(eval)     # parse list strings → Python lists
df['Action'] = df['Action'].astype(int)         # ensure integer labels

# Normalize StateVec
scaler = MinMaxScaler()
state_vecs = scaler.fit_transform(np.vstack(df["StateVec"])).astype(np.float32)
labels = df["Action"].tolist()

# ✅ Step 4: Custom offline environment with reward shaping
class CloudSchedulerOfflineDQNEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.states = state_vecs
        self.labels = labels
        self.current_step = 0
        self.n_tasks = len(self.states)

        # Observation: 5D normalized features; Action: 3 discrete clouds
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self.states[self.current_step], {}

    def step(self, action):
        correct_action = self.labels[self.current_step]

        # ✅ Reward shaping: +1 if correct; graded penalty otherwise
        if action == correct_action:
            reward = 1.0
        else:
            reward = -0.25 * abs(action - correct_action)

        # Advance pointer and determine termination
        self.current_step += 1
        done = self.current_step >= self.n_tasks
        obs = self.states[self.current_step] if not done else np.zeros(5, dtype=np.float32)

        terminated = bool(done)
        truncated = False
        info = {}
        return obs, float(reward), terminated, truncated, info

# ✅ Step 5: Setup environment
env = CloudSchedulerOfflineDQNEnv()
check_env(env)  # sanity-check spaces & API

# ✅ Step 6: Train DQN model
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0005,
    exploration_fraction=0.3,
    buffer_size=10000,
    batch_size=64,
    gamma=0.99,
    train_freq=1,
    target_update_interval=100
)

model.learn(total_timesteps=50000)
model.save("dqn_v1")  # 🔁 Saves dqn_v1.zip
print("✅ DQN training complete. Model saved as dqn_v1.zip")
