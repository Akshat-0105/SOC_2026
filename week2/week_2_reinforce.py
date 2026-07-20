import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(state_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        """state: tensor (batch_size, state_dim)-> probs (batch_size,
        action_dim)"""
        logits = self.net(state)
        probs = torch.softmax(logits, dim=-1)
        return probs
    
def run_episode(env, policy, render=False, gamma=0.99):
    states, actions, rewards, log_probs = [], [], [], []

    obs, info = env.reset()
    done = False

    while not done:
        state = torch.from_numpy(obs).float().unsqueeze(0)

        # Pass the current state through the policy network to obtain
        # the probability of taking each possible action.
        probs = policy(state) 

        # Create a categorical probability distribution from the
        # action probabilities produced by the policy network.
        dist = torch.distributions.Categorical(probs=probs)

        # Sample an action from the probability distribution.
        # This allows the agent to explore different actions.
        action = dist.sample()
        
        if render:
            env.render()
        next_obs, reward, terminated, truncated, info = env.step(action.item()
        )
        done = terminated or truncated
        states.append(obs)
        actions.append(action.item())
        rewards.append(reward)
        log_probs.append(dist.log_prob(action))
        obs = next_obs
    
    # Compute the discounted return for every time step by
    # traversing the rewards in reverse order.
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    log_probs = torch.stack(log_probs)
    return states, actions, rewards, returns, log_probs

def train_reinforce(
    env_name="CartPole-v1",
    hidden_dim=64,
    learning_rate=1e-2,
    gamma=0.99,
    num_episodes=500,
):
    env = gym.make(env_name, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = PolicyNetwork(state_dim, hidden_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    all_episode_returns = []
    for episode in range(num_episodes):
        states, actions, rewards, returns, log_probs = run_episode(
        env, policy, render=False, gamma=gamma
        )
        returns_mean = returns.mean()
        returns_std = returns.std() + 1e-8
        normalized_returns = (returns- returns_mean) / returns_std

        # REINFORCE loss:
        # Actions that lead to higher returns should become more likely.
        # The negative sign is used because PyTorch minimizes the loss,
        # while reinforcement learning tries to maximize expected reward.
        loss =-(log_probs * normalized_returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        episode_return = sum(rewards)
        all_episode_returns.append(episode_return)
        if (episode + 1) % 10 == 0:
            avg_last_10 = np.mean(all_episode_returns[-10:])
            print(
            f"Episode {episode + 1:4d} | "
            f"Return: {episode_return:6.1f} | "
            f"Avg last 10: {avg_last_10:6.1f}"
            )
    env.close()
    return all_episode_returns, policy
import matplotlib.pyplot as plt
import os

def plot_returns(returns, window=10, filename="plots/cartpole_rewards.png"):
    os.makedirs("plots", exist_ok=True)

    returns = np.array(returns)
    x = np.arange(len(returns))

    if len(returns) >= window:
        sma = np.convolve(
            returns,
            np.ones(window) / window,
            mode="valid"
        )
    else:
        sma = returns

    plt.figure(figsize=(8,4))
    plt.plot(x, returns, alpha=0.3, label="Episode return")
    plt.plot(np.arange(len(sma)), sma, label="Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":

    returns, trained_policy = train_reinforce()

    np.save("reinforce_returns.npy", returns)

    print("Saved reinforce_returns.npy")