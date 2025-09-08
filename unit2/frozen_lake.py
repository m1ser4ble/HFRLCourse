import numpy as np
import gymnasium as gym
import os


# from tqdm.notebook import tqdm
from qlearning import *

# Create the FrozenLake-v1 environment using 4x4 map and non-slippery version and render_mode="rgb_array"
# env = gym.make()  # TODO use the correct parameters
env = gym.make(
    "FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="rgb_array"
)


desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
gym.make("FrozenLake-v1", desc=desc, is_slippery=True)

# We create our environment with gym.make("<name_of_the_environment>")- `is_slippery=False`: The agent always moves in the intended direction due to the non-slippery nature of the frozen lake (deterministic).
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space", env.observation_space)
print("Sample observation", env.observation_space.sample())  # Get a random observation


print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample())  # Take a random action


state_space = env.observation_space.n
print("There are ", state_space, " possible states")

action_space = env.action_space.n
print("There are ", action_space, " possible actions")


Qtable_frozenlake = initialize_q_table(state_space, action_space)


# Training parameters
n_training_episodes = 10000  # Total training episodes
learning_rate = 0.7  # Learning rate

# Evaluation parameters
n_eval_episodes = 100  # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"  # Name of the environment
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate
eval_seed = []  # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob


Qtable_frozenlake = train(
    n_training_episodes,
    min_epsilon,
    max_epsilon,
    gamma,
    learning_rate,
    decay_rate,
    env,
    max_steps,
    Qtable_frozenlake,
)


# Evaluate our Agent
mean_reward, std_reward = evaluate_agent(
    env, max_steps, n_eval_episodes, Qtable_frozenlake, eval_seed
)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")


model = {
    "env_id": env_id,
    "max_steps": max_steps,
    "n_training_episodes": n_training_episodes,
    "n_eval_episodes": n_eval_episodes,
    "eval_seed": eval_seed,
    "learning_rate": learning_rate,
    "gamma": gamma,
    "max_epsilon": max_epsilon,
    "min_epsilon": min_epsilon,
    "decay_rate": decay_rate,
    "qtable": Qtable_frozenlake,
}

username = "hung3r"  # FILL THIS
repo_name = "q-FrozenLake-v1-4x4-noSlippery"
push_to_hub(env_id=env_id, repo_id=f"{username}/{repo_name}", model=model, env=env)
