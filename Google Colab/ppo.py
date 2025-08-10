# ============================================================
# Title: PPO Training on Cleaned 5-Feature Dataset
# Purpose:
#   - Load a cleaned dataset with 5D StateVec/NextState
#   - Normalize features
#   - Train a Stable-Baselines3 PPO agent in a simple Gymnasium env
#   - Save the pretrained model for later inference
# Notes:
#   - Expects 'ppo_training_dataset_cleaned_5f.csv' with columns:
#     StateVec (list[5]), NextState (list[5]), Reward, Done, Action(optional), SLAMet(optional)
#   - Uses shimmy to bridge Gymnasium with Stable-Baselines3.
#   - Action space is Discrete(3): e.g., clouds {0,1,2}.
# ============================================================

# ✅ Step 1: Install Required Libraries
#!pip install stable-baselines3[extra] gymnasium pandas scikit-learn --quiet
#!pip install "shimmy>=2.0"

# ✅ Step 2: Import Required Modules
import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import ast

# ✅ Step 3: Load and Parse the Cleaned Dataset
df = pd.read_csv("/content/ppo_training_dataset_cleaned_5f.csv")
df['StateVec'] = df['StateVec'].apply(ast.literal_eval)
df['NextState'] = df['NextState'].apply(ast.literal_eval)

# ✅ Step 4: Normalize StateVec and NextState
scaler = MinMaxScaler()
all_states = np.vstack(df['StateVec'].tolist() + df['NextState'].tolist())  # fit on combined
scaler.fit(all_states)

df['StateVec'] = df['StateVec'].apply(lambda x: scaler.transform([x])[0])
df['NextState'] = df['NextState'].apply(lambda x: scaler.transform([x])[0])

# ✅ Step 5: Define the PPO-Compatible Gym Environment
class PPOCloudEnv(gym.Env):
    def __init__(self, dataframe):
        super(PPOCloudEnv, self).__init__()
        self.df = dataframe.reset_index(drop=True)
        self.current_step = 0
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)  # 5D normalized
        self.action_space = spaces.Discrete(3)  # 3 clouds

    def reset(self, seed=None, options=None):
        self.current_step = 0
        state = np.array(self.df.loc[self.current_step, 'StateVec'], dtype=np.float32)
        return state, {}  # Gymnasium API: (obs, info)

    def step(self, action):
        row = self.df.loc[self.current_step]
        reward = row['Reward']           # scalar reward from dataset
        done = row['Done']               # per-row termination flag (may be False)
        next_state = np.array(row['NextState'], dtype=np.float32)

        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True                  # end episode at end of dataset

        return next_state, reward, done, False, {}  # (obs, reward, terminated, truncated, info)

# ✅ Step 6: Initialize PPO and Train
env = PPOCloudEnv(df)
vec_env = DummyVecEnv([lambda: env])                     # wrap for SB3
model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_log")
model.learn(total_timesteps=20000)                       # training steps

# ✅ Step 7: Save the Trained PPO Model
model.save("ppo_v2.zip")
print("✅ PPO Model saved as 'ppo_v2.zip'")

# ✅ Optional: Download the Model File
from google.colab import files
files.download("ppo_v2.zip")  # NOTE: Filename differs from saved name above; adjust if needed.
