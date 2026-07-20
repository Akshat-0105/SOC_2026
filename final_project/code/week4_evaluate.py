# week4_evaluate.py
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from week4_trading_env_v2 import TradingEnvV2

def run_episode(env, model=None):
    """Run one episode. model=None means random agent."""
    obs, _ = env.reset()
    rewards = []
    done = False
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            # During evaluation we use deterministic=True so the agent always
            # chooses its best learned action instead of exploring randomly.
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return rewards

def buy_and_hold_episode(env):
    """Always stay long (action=0). Returns per-step rewards."""
    obs, _ = env.reset()
    rewards = []
    done = False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(0)
        rewards.append(reward)
        done = terminated or truncated
    return rewards
    
def average_cumulative_pnl(env, n_episodes=50,
        model=None, strategy="random"):
    """Return the mean cumulative P&L curve over n_episodes."""
    all_cumulative = []
    for _ in range(n_episodes):
        if strategy == "buy_and_hold":
            step_rewards = buy_and_hold_episode(env)
        else:
            step_rewards = run_episode(env, model=model)
        # np.cumsum converts the per-step rewards into cumulative profit
        # and loss (P&L) over the entire episode.
        all_cumulative.append(np.cumsum(step_rewards))
    min_len = min(len(c) for c in all_cumulative)
    stacked = np.array([c[:min_len] for c in all_cumulative])
    return stacked.mean(axis=0)

def plot_pnl(pnl_trained, pnl_random, pnl_bnh,
        filename="plots/cumulative_pnl.png"):
    os.makedirs("plots", exist_ok=True)
    steps = np.arange(len(pnl_trained))
    plt.figure(figsize=(9, 4))
    plt.plot(steps, pnl_trained, color="steelblue",
        label="PPO (trained)")
    plt.plot(steps, pnl_random, color="darkorange",
        label="Random agent")
    plt.plot(steps, pnl_bnh,
        color="green",
        label="Buy and hold")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Step within episode")
    plt.ylabel("Cumulative P&L (mean over 50 episodes)")
    plt.title("Agent comparison on TradingEnvV2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved to {filename}")

if __name__ == "__main__":
    env = TradingEnvV2()
    
    print("Training PPO on TradingEnvV2 ...")
    model = PPO(
        "MlpPolicy", env, verbose=0,
        learning_rate=3e-4, n_steps=512,
        batch_size=64, gamma=0.99,
    )
    model.learn(total_timesteps=100_000)
    print("Training complete.")
    
    env_eval = TradingEnvV2()
    pnl_trained = average_cumulative_pnl(
        env_eval, n_episodes=50, model=model, strategy="trained")
    pnl_random = average_cumulative_pnl(
        env_eval, n_episodes=50, model=None, strategy="random")
    pnl_bnh = average_cumulative_pnl(
        env_eval, n_episodes=50, model=None, strategy="buy_and_hold")
    print(f"Final cumulative P&L (mean over 50 episodes):")
    print(f" PPO trained: {pnl_trained[-1]:.2f}")
    print(f" Random agent: {pnl_random[-1]:.2f}")
    print(f" Buy and hold: {pnl_bnh[-1]:.2f}")
    plot_pnl(pnl_trained, pnl_random, pnl_bnh)