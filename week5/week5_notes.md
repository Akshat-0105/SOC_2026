# Week 5 Notes

## Task 1
- Implemented `TradingEnvV3` by replacing the independent return process with an AR(1) process:
  \[
  r_t = \rho r_{t-1} + \epsilon_t
  \]
- Verified the environment using `check_env()` and random rollouts.

## Task 2
Measured autocorrelation of generated returns:

| rho | Measured autocorrelation |
|-----:|-------------------------:|
| 0.00 | ≈ 0.000 |
| 0.50 | ≈ 0.501 |
| -0.50 | ≈ -0.501 |

The measured values closely matched the configured values, confirming that the AR(1) implementation was correct.

## Task 3
- Trained PPO for 200,000 timesteps.
- Verified that the observation included the latest return and current position.
- The learned policy did not consistently outperform the baselines. Across runs, the agent often converged to either staying flat or holding a single position throughout the episode, resulting in cumulative P&L close to zero or below the buy-and-hold baseline.

## Task 4

### Sweep 1 (transaction_cost = 0.1)

| rho | Mean cumulative P&L |
|----:|--------------------:|
| 0.00 | +0.00 |
| 0.25 | +0.00 |
| 0.50 | +0.00 |

### Sweep 2 (rho = 0.25)

| transaction_cost | Mean cumulative P&L |
|-----------------:|--------------------:|
| 0.00 | +0.00 |
| 0.10 | -2.17 |
| 0.30 | +0.00 |

### Observations

- The AR(1) process correctly generated positive and negative autocorrelation.
- PPO training showed significant variability and frequently converged to a conservative policy with little or no trading.
- Increasing transaction cost generally reduced performance, while the strongest positive signal did not consistently translate into higher profits for this particular set of training runs.