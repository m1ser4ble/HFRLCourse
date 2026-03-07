import numpy as np

from collections import deque
from common import reinforce, push_to_hub, evaluate_agent
import matplotlib.pyplot as plt
# %matplotlib inline

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
# import gym
# import gym_pygame
import gymnasium as gym

# Hugging Face Hub
from huggingface_hub import (
    notebook_login,
)  # To log to our Hugging Face account to be able to upload models to the Hub.
import imageio

device = torch.device("mps")

env_id = "CartPole-v1"
# Create the env
env = gym.make(env_id, render_mode="rgb_array")

# Create the evaluation env
eval_env = gym.make(env_id, render_mode="rgb_array")

# Get the state space and action space
s_size = env.observation_space.shape[0]
a_size = env.action_space.n
print(env.action_space.n)


# model definition
class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        # a_size : action size
        # h_size : hidden size
        # s_size : state size
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 1000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": int(a_size),
}


# Create policy and place it to the device
cartpole_policy = Policy(
    cartpole_hyperparameters["state_space"],
    cartpole_hyperparameters["action_space"],
    cartpole_hyperparameters["h_size"],
).to(device)
cartpole_optimizer = optim.Adam(
    cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"]
)

scores = reinforce(
    env,
    cartpole_policy,
    cartpole_optimizer,
    cartpole_hyperparameters["n_training_episodes"],
    cartpole_hyperparameters["max_t"],
    cartpole_hyperparameters["gamma"],
    100,
)


evaluate_agent(
    eval_env,
    cartpole_hyperparameters["max_t"],
    cartpole_hyperparameters["n_evaluation_episodes"],
    cartpole_policy,
)


from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.repocard import metadata_eval_result, metadata_save

from pathlib import Path
import datetime
import json
import imageio

import tempfile

import os


repo_id = "hung3r/Reinforce-Cartpole-v1"  # TODO Define your repo id {username/Reinforce-{model-id}}
push_to_hub(
    repo_id,
    cartpole_policy,  # The model we want to save
    cartpole_hyperparameters,  # Hyperparameters
    eval_env,  # Evaluation environment
    video_fps=30,
)
