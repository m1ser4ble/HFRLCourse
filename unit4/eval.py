import matplotlib.pyplot as plt

import torch
import gymnasium as gym
from ple import PLE
from ple.games.pixelcopter import Pixelcopter
import numpy as np

# Load the trained model
# Note: Adjust path based on your model file structure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("pixelcopter_reinforce_model.pth", map_location=device)
model.eval()


# Create the environment
def create_pixelcopter_env():
    game = Pixelcopter()
    env = PLE(game, fps=30, display=True)  # Set display=False for headless
    return env


# Initialize environment
env = create_pixelcopter_env()
env.init()


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


def evaluate_agent_detailed(model, env, episodes=50):
    """Detailed evaluation with statistics and visualization"""
    rewards = []
    episode_lengths = []

    for episode in range(episodes):
        env.reset_game()
        episode_reward = 0
        steps = 0

        while not env.game_over():
            state = preprocess_state(env.getScreenRGB())
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action_probs = model(state_tensor)
                action = torch.multinomial(action_probs, 1).item()

            reward = env.act(action)
            episode_reward += reward
            steps += 1

        rewards.append(episode_reward)
        episode_lengths.append(steps)

        if (episode + 1) % 10 == 0:
            print(f"Episodes {episode + 1}/{episodes} completed")

    # Statistical analysis
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    median_reward = np.median(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)

    mean_length = np.mean(episode_lengths)

    print(f"\n--- Evaluation Results ---")
    print(f"Episodes: {episodes}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Median Reward: {median_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.1f} steps")

    # Visualization
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.axhline(
        y=mean_reward, color="r", linestyle="--", label=f"Mean: {mean_reward:.2f}"
    )
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(rewards, bins=20, alpha=0.7)
    plt.axvline(
        x=mean_reward, color="r", linestyle="--", label=f"Mean: {mean_reward:.2f}"
    )
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return {
        "rewards": rewards,
        "episode_lengths": episode_lengths,
        "stats": {
            "mean": mean_reward,
            "std": std_reward,
            "median": median_reward,
            "max": max_reward,
            "min": min_reward,
        },
    }


# Run detailed evaluation
results = evaluate_agent_detailed(model, env, episodes=100)
