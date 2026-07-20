# week5_train_and_evaluate.py
import numpy as np
from stable_baselines3 import PPO
from week5_trading_env_v3 import TradingEnvV3
from week4_evaluate import average_cumulative_pnl, plot_pnl

if __name__ == "__main__":
    env = TradingEnvV3(rho=0.5, transaction_cost=0.1)
    
    print("Training PPO on the momentum market (rho = 0.5) ...")
    model = PPO(
        "MlpPolicy", env, verbose=0,
        learning_rate=3e-4, n_steps=512,
        batch_size=64, gamma=0.99,
    )
    model.learn(total_timesteps=200_000)
    print("Training complete.")
    
    env_eval = TradingEnvV3(rho=0.5, transaction_cost=0.1)
    pnl_trained = average_cumulative_pnl(
        env_eval, n_episodes=50, model=model, strategy="trained")
    pnl_random = average_cumulative_pnl(
        env_eval, n_episodes=50, model=None, strategy="random")
    pnl_bnh = average_cumulative_pnl(
        env_eval, n_episodes=50, model=None, strategy="buy_and_hold")
    print("Final cumulative P&L (mean over 50 episodes):")
    print(f" PPO trained: {pnl_trained[-1]:+8.2f}")
    print(f" Random agent: {pnl_random[-1]:+8.2f}")
    print(f" Buy and hold: {pnl_bnh[-1]:+8.2f}")
    
    plot_pnl(pnl_trained, pnl_random, pnl_bnh,
        filename="plots/cumulative_pnl_momentum.png")
    
    print("\nObservation check")
    print("-----------------")
    
    obs, _ = env_eval.reset()
    print("After reset:", obs)
    
    action = env_eval.action_space.sample()
    obs, reward, terminated, truncated, _ = env_eval.step(action)
    print("After one step:", obs)

    print("\nPolicy inspection")
    print("-----------------")

    obs, _ = env_eval.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env_eval.step(action)
        print(f"last return {obs[-2]:+.4f} -> position {obs[-1]:+.0f}")
        done = terminated or truncated