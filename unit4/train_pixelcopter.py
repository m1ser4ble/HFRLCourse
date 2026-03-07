import torch
from torch import nn
import gymnasium as gym
from ple import PLE
from ple.games.pixelcopter import Pixelcopter
import numpy as np

# Load the trained model
# Note: Adjust path based on your model file structure
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
# model = torch.load("pixelcopter_reinforce_model.pth", map_location=device)
# model.eval()


# Create the environment
def create_pixelcopter_env():
    game = Pixelcopter()
    env = PLE(
        game,
        fps=30,
    )  # Set display=False for headless
    return env


# Initialize environment
env = create_pixelcopter_env()
env.init()

breakpoint()

model = Policy()


# Run trained agent
def run_agent(model, env, episodes=5):
    total_rewards = []

    for episode in range(episodes):
        env.reset_game()
        episode_reward = 0

        while not env.game_over():
            # Get current state
            state = env.getScreenRGB()  # or env.getGameState() if using features
            state = preprocess_state(state)  # Apply your preprocessing

            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Get action probabilities
            with torch.no_grad():
                action_probs = model(state_tensor)
                action = torch.multinomial(action_probs, 1).item()

            # Execute action (0: do nothing, 1: thrust)
            reward = env.act(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nAverage Performance: {mean_reward:.2f} ± {std_reward:.2f}")

    return total_rewards


# Preprocessing function (adjust based on your model's input requirements)
def preprocess_state(state):
    """
    Preprocess the game state for the neural network
    This should match the preprocessing used during training
    """
    if isinstance(state, np.ndarray) and len(state.shape) == 3:
        # If using image input
        state = np.transpose(state, (2, 0, 1))  # Channel first
        state = state / 255.0  # Normalize pixels
        return state.flatten()  # or keep as image depending on model
    else:
        # If using game state features
        return np.array(list(state.values()))


# Run the agent
rewards = run_agent(model, env, episodes=10)
