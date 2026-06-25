import numpy as np
import matplotlib.pyplot as plt
import os

def moving_average(x, window=10):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")

def plot_comparison(reinforce_returns, ppo_returns,
    window=10,
    filename="plots/reinforce_vs_ppo.png"):
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    # REINFORCE
    r = np.array(reinforce_returns)
    ax.plot(r, alpha=0.2, color="steelblue")
    ax.plot(moving_average(r, window), color="steelblue",
    label=f"REINFORCE ({window}-ep avg)")
    # PPO
    p = np.array(ppo_returns)
    ax.plot(p, alpha=0.2, color="darkorange")
    ax.plot(moving_average(p, window), color="darkorange",
    label=f"PPO ({window}-ep avg)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("REINFORCE vs PPO on CartPole-v1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    # Load your saved returns--- adjust filenames as needed
    reinforce_returns = np.load("reinforce_returns.npy").tolist()
    ppo_returns = np.load("ppo_returns.npy").tolist()
    plot_comparison(reinforce_returns, ppo_returns)