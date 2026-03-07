import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from ple.games.pixelcopter import Pixelcopter
from ple import PLE


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


# Create the environment
def create_pixelcopter_env():
    game = Pixelcopter()
    env = PLE(
        game,
        fps=30,
    )  # Set display=False for headless
    return env


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


class REINFORCEAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state)
        breakpoint()
        action = torch.multinomial(probs, 1)

        self.saved_log_probs.append(torch.log(probs.squeeze(0)[action]))
        return action.item()

    def update_policy(self, gamma=0.99):
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0

        for r in reversed(self.rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-8
        )

        # Calculate policy loss
        policy_loss = []
        for log_prob, reward in zip(self.saved_log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)

        # Update policy
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # Clear episode data
        self.saved_log_probs.clear()
        self.rewards.clear()

        return policy_loss.item()


def train_agent(episodes=20000):
    env = create_pixelcopter_env()
    env.init()

    # Determine state size based on your preprocessing
    state_size = len(preprocess_state(env.getScreenRGB()))  # Adjust as needed
    action_size = 2  # do nothing, thrust

    agent = REINFORCEAgent(state_size, action_size)

    scores = deque(maxlen=100)

    for episode in range(episodes):
        env.reset_game()
        episode_reward = 0

        while not env.game_over():
            state = preprocess_state(env.getScreenRGB())
            action = agent.select_action(state)

            print(f"episode {episode} action : {action}")
            reward = env.act(action)
            agent.rewards.append(reward)
            episode_reward += reward

        # Update policy after each episode
        loss = agent.update_policy()
        scores.append(episode_reward)

        if episode % 100 == 0:
            avg_score = np.mean(scores)
            print(
                f"Episode {episode}, Average Score: {avg_score:.2f}, Loss: {loss:.4f}"
            )

    # Save the trained model
    torch.save(agent.policy_net, "pixelcopter_reinforce_model.pth")
    return agent


# Train a new agent
trained_agent = train_agent()
