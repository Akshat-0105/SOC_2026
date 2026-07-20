import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import yfinance as yf
from stable_baselines3 import PPO
from week6_historical_env import HistoricalTradingEnv
from upgraded_trading_env import UpgradedTradingEnv

# --- Evaluation Metric Helpers ---
def sharpe_ratio(step_rewards, periods_per_year=252):
    r = np.asarray(step_rewards, dtype=float)
    if r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))

def max_drawdown(step_rewards):
    equity = np.cumsum(np.asarray(step_rewards, dtype=float))
    peak = np.maximum.accumulate(equity)
    return float((peak - equity).max())

# --- Data Engine Pipeline ---
def get_market_data(ticker, start="2015-01-01", end="2024-12-31"):
    print(f"\nFetching historical data for: {ticker}...")
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    close = data["Close"].squeeze().dropna()
    prices = close.to_numpy(dtype=float)
    returns = np.diff(np.log(prices))
    
    # Calculate Summary Stats
    lag1 = np.corrcoef(returns[:-1], returns[1:])[0, 1]
    print(f"[{ticker} Statistics]")
    print(f"  Trading Days: {len(returns)}")
    print(f"  Mean Daily Return: {returns.mean():+.5f}")
    print(f"  Std of Daily Returns: {returns.std():.5f}")
    print(f"  Lag-1 Autocorrelation: {lag1:+.4f}")
    
    return returns

# --- Strategy Evaluator Engine ---
def evaluate_strategy(env, model=None, strategy="trained", n_episodes=100, seed=42):
    all_pnl_curves = []
    all_sharpes = []
    all_drawdowns = []
    
    # Anchor seed for baseline comparisons
    env.unwrapped.np_random = np.random.default_rng(seed)
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        rewards = []
        done = False
        while not done:
            if strategy == "random":
                action = env.action_space.sample()
            elif strategy == "buy_and_hold":
                action = 0 # Standard long position anchor
            else:
                action, _ = model.predict(obs, deterministic=True)
                
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
            
        all_pnl_curves.append(np.cumsum(rewards))
        all_sharpes.append(sharpe_ratio(rewards))
        all_drawdowns.append(max_drawdown(rewards))
        
    min_len = min(len(c) for c in all_pnl_curves)
    mean_pnl_curve = np.array([c[:min_len] for c in all_pnl_curves]).mean(axis=0)
    
    return mean_pnl_curve, np.mean(all_sharpes), np.mean(all_drawdowns)

# --- Executive Pipeline Controller ---
def run_full_study():
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # 1. Market Targets
    markets = ["SPY", "RELIANCE.NS"]
    seeds = [42, 101, 2026]
    budget = 100_000
    
    for ticker in markets:
        print("\n" + "="*50)
        print(f"STARTING COMPREHENSIVE EXPERIMENT FOR {ticker}")
        print("="*50)
        
        returns = get_market_data(ticker)
        
        # 80/20 Chronological Split boundary calculation
        split_idx = int(len(returns) * 0.80)
        train_returns = returns[:split_idx]
        test_returns = returns[split_idx:]
        
        # Instantiate Evaluation Benchmark Baseline Environments
        test_env_plain = HistoricalTradingEnv(test_returns)
        test_env_upgraded = UpgradedTradingEnv(test_returns)
        
        # Compute Static Baselines once
        print("\nCalculating benchmarks (Random and Buy-and-Hold)...")
        pnl_rand, sr_rand, dd_rand = evaluate_strategy(test_env_upgraded, strategy="random")
        pnl_bnh, sr_bnh, dd_bnh = evaluate_strategy(test_env_upgraded, strategy="buy_and_hold")
        
        # Data storage matrices for seed loops
        plain_pnl_list, plain_sr_list, plain_dd_list = [], [], []
        upgraded_pnl_list, upgraded_sr_list, upgraded_dd_list = [], [], []
        
        # 2. Multi-Seed Grid Execution Loop
        for seed in seeds:
            print(f"\n--- Processing Seed: {seed} ---")
            
            # Setup train environments with separate random seeds
            train_env_plain = HistoricalTradingEnv(train_returns)
            train_env_upgraded = UpgradedTradingEnv(train_returns)
            
            # Train Baseline PPO Agent
            print(f"Training Baseline PPO Agent...")
            model_plain = PPO("MlpPolicy", train_env_plain, verbose=0, seed=seed)
            model_plain.learn(total_timesteps=budget)
            model_plain.save(f"models/ppo_plain_{ticker}_{seed}")
            
            # Evaluate Baseline PPO Agent
            pnl_p, sr_p, dd_p = evaluate_strategy(test_env_plain, model=model_plain, strategy="trained")
            plain_pnl_list.append(pnl_p)
            plain_sr_list.append(sr_p)
            plain_dd_list.append(dd_p)
            
            # Train Upgraded PPO Agent
            print(f"Training Upgraded PPO Agent...")
            model_upgraded = PPO("MlpPolicy", train_env_upgraded, verbose=0, seed=seed)
            model_upgraded.learn(total_timesteps=budget)
            model_upgraded.save(f"models/ppo_upgraded_{ticker}_{seed}")
            
            # Evaluate Upgraded PPO Agent
            pnl_u, sr_u, dd_u = evaluate_strategy(test_env_upgraded, model=model_upgraded, strategy="trained")
            upgraded_pnl_list.append(pnl_u)
            upgraded_sr_list.append(sr_u)
            upgraded_dd_list.append(dd_u)
            
        # 3. Calculate Aggregated Run Statistics (Mean ± Max Spread)
        def compute_metrics_summary(metric_list):
            arr = np.array(metric_list)
            mean_val = arr.mean()
            spread = (arr.max() - arr.min()) / 2.0
            return f"{mean_val:+.2f} ± {spread:.2f}"
            
        print(f"\n{'='*10} TEST SET RESULTS SUMMARY FOR {ticker} {'='*10}")
        print(f"Strategy          | Cumulative P&L  | Sharpe Ratio   | Max Drawdown")
        print(f"-" * 70)
        print(f"Buy & Hold        | {pnl_bnh[-1]:+.2f}          | {sr_bnh:.4f}         | {dd_bnh:.2f}")
        print(f"Random Agent      | {pnl_rand[-1]:+.2f}          | {sr_rand:.4f}         | {dd_rand:.2f}")
        print(f"Baseline PPO Agent| {compute_metrics_summary([p[-1] for p in plain_pnl_list])}   | {compute_metrics_summary(plain_sr_list)}   | {compute_metrics_summary(plain_dd_list)}")
        print(f"Upgraded PPO Agent| {compute_metrics_summary([u[-1] for u in upgraded_pnl_list])}   | {compute_metrics_summary(upgraded_sr_list)}   | {compute_metrics_summary(upgraded_dd_list)}")
        
        # 4. Generate Performance Visualization Plots
        plt.figure(figsize=(10, 5))
        steps = np.arange(len(pnl_bnh))
        plt.plot(steps, pnl_bnh, label="Buy and Hold", color="green", linestyle="--")
        plt.plot(steps, pnl_rand, label="Random Agent", color="orange", alpha=0.6)
        
        # Plot mean performance curve across all seed configurations
        mean_plain_pnl = np.array(plain_pnl_list).mean(axis=0)
        mean_upgraded_pnl = np.array(upgraded_pnl_list).mean(axis=0)
        
        plt.plot(steps, mean_plain_pnl, label="Baseline PPO (Mean)", color="crimson")
        plt.plot(steps, mean_upgraded_pnl, label="Upgraded PPO (Mean)", color="blue", linewidth=2)
        
        plt.axhline(0, color="black", linestyle=":", alpha=0.5)
        plt.title(f"Test Set Evaluation Comparison: {ticker}")
        plt.xlabel("Execution Timestep within Window")
        plt.ylabel("Cumulative P&L Profile")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plots/{ticker}_evaluation_comparison.png")
        plt.close()
        print(f"\nSaved tracking visualization asset to plots/{ticker}_evaluation_comparison.png")

if __name__ == "__main__":
    run_full_study()