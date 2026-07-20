import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# week4_train_ppo_trading.py
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from week3_trading_env import ToyTradingEnv

class EpisodeRewardLogger(BaseCallback):
    """Records the total reward of each completed episode."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current_reward = 0.0
    def _on_step(self)-> bool:
        self._current_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_reward)
        self._current_reward = 0.0
        return True
def train_ppo_on_trading(total_timesteps=100_000):
    env = ToyTradingEnv()
    callback = EpisodeRewardLogger()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=512,
        # Number of environment steps collected before each PPO policy update.
        batch_size=64, # Number of samples used in each mini-batch during optimization.
        gamma=0.99,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    env.close()
    return model, callback.episode_rewards

def moving_average(x, window=20):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")

def plot_rewards(episode_rewards,
    filename="plots/ppo_trading_rewards.png"):
    os.makedirs("plots", exist_ok=True)
    rewards = np.array(episode_rewards)
    plt.figure(figsize=(9, 4))
    plt.plot(rewards, alpha=0.25, color="steelblue",
    label="Episode reward")
    plt.plot(moving_average(rewards), color="steelblue",
    label="20-ep moving avg")
    plt.axhline(0, color="gray", linestyle="--",
    linewidth=0.8, label="Zero")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO on ToyTradingEnv")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved to {filename}")

if __name__ == "__main__":
    model, rewards = train_ppo_on_trading(total_timesteps=100_000)
    plot_rewards(rewards)
    np.save("ppo_trading_returns.npy", np.array(rewards))
    print(f"Final 20-episode average: {np.mean(rewards[-20:]):.2f}")