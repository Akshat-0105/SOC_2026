# week6_train_test.py
import numpy as np
from stable_baselines3 import PPO
from week6_historical_env import HistoricalTradingEnv
from week4_evaluate import average_cumulative_pnl, plot_pnl

def chronological_split(returns, train_fraction=0.8):
    """First 80% of days-> training, last 20%-> testing.
    NEVER shuffle before splitting-- see this week's resources."""
    # Never shuffle time-series data before splitting because it would leak
    # future information into the training set, producing an unrealistically
    # optimistic evaluation (lookahead bias).
    split = int(len(returns) * train_fraction)
    return returns[:split], returns[split:]

if __name__ == "__main__":
    returns = np.load("data/returns.npy")
    train_returns, test_returns = chronological_split(returns)
    print(f"Train: {len(train_returns)} days | "
        f"Test: {len(test_returns)} days")
    
    train_env = HistoricalTradingEnv(train_returns)
    print("Training PPO on the training years ...")
    model = PPO("MlpPolicy", train_env, verbose=0,
        learning_rate=3e-4, n_steps=512,
        batch_size=64, gamma=0.99)
    model.learn(total_timesteps=100_000)
    print("Training complete.")
    # Evaluate on data the agent HAS seen (train)
    # and on data it has NOT seen (test)
    train_eval = HistoricalTradingEnv(train_returns)
    test_eval = HistoricalTradingEnv(test_returns)
    pnl_train = average_cumulative_pnl(
        train_eval, n_episodes=50, model=model, strategy="trained")
    pnl_test = average_cumulative_pnl(
        test_eval, n_episodes=50, model=model, strategy="trained")
    pnl_random = average_cumulative_pnl(
        test_eval, n_episodes=50, model=None, strategy="random")
    pnl_bnh = average_cumulative_pnl(
        test_eval, n_episodes=50, model=None, strategy="buy_and_hold")
    
    print("Final mean cumulative P&L over 50 episodes:")
    print(f" PPO on TRAIN data: {pnl_train[-1]:+8.2f}")
    print(f" PPO on TEST data: {pnl_test[-1]:+8.2f}")
    print(f" Random (test): {pnl_random[-1]:+8.2f}")
    print(f" Buy & hold (test): {pnl_bnh[-1]:+8.2f}")
    
    plot_pnl(pnl_test, pnl_random, pnl_bnh,
        filename="plots/test_cumulative_pnl.png")