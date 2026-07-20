# Week 6 Notes

## 1. Compare the real daily returns with the synthetic returns from Week 5.

The real data had a mean daily return of +0.00049 and a standard deviation of 0.01114, which is very close to the synthetic daily volatility of 0.01 used in Week 5. However, the lag-1 autocorrelation was -0.1163, indicating very weak (slightly negative) correlation between consecutive daily returns. In contrast, the synthetic environment allowed us to generate data with controlled autocorrelation (e.g., ρ = 0.5), making predictable momentum much easier to learn.

---

## 2. Why must we split the data chronologically instead of randomly?

Financial time series are ordered in time. Randomly shuffling the data would leak future information into the training set, resulting in lookahead bias and unrealistic performance. A chronological split ensures the agent only learns from past data and is evaluated on unseen future data, similar to real-world trading.

---

## 3. Compare PPO performance on the training and test sets.

### Run 1

- PPO (Train): +5.16
- PPO (Test): +8.60

### Run 2

- PPO (Train): +0.00
- PPO (Test): +0.00

### Run 3

- PPO (Train): +3.69
- PPO (Test): +9.54

The PPO agent showed considerable variability across different training runs. There was no consistent pattern where training performance was much higher than testing performance. In two runs, the test performance was actually higher than the training performance, suggesting no obvious overfitting.

---

## 4. Compare PPO with the random policy.

Across all three runs, the random policy consistently produced negative cumulative P&L:

- Run 1: -6.86
- Run 2: -8.86
- Run 3: -7.77

The PPO agent outperformed the random policy in every run, showing that it learned a strategy better than random actions.

---

## 5. Compare PPO with buy-and-hold.

Buy-and-hold remained the strongest baseline throughout the experiments.

| Run | PPO Test | Buy & Hold |
|------|----------|------------|
| 1 | +8.60 | +9.90 |
| 2 | +0.00 | +9.52 |
| 3 | +9.54 | +10.36 |

Although PPO learned profitable policies in two runs, it did not outperform the simple buy-and-hold strategy.

---

## 6. Why is buy-and-hold such a strong baseline?

The S&P 500 has a long-term upward trend, so simply remaining invested captures the overall market growth. Since daily returns contain only weak short-term predictability, buy-and-hold is difficult for a reinforcement learning agent to consistently outperform.

---

## 7. Did the agent overfit?

There is no strong evidence of overfitting. Overfitting would typically appear as very high training performance combined with poor testing performance. Instead, the PPO agent showed similar or even better performance on the test set in successful runs, while the main observation was variability between different training runs.

---

## 8. Why is evaluating on unseen future data important?

Evaluating on unseen future data provides a realistic estimate of how the trading strategy would perform in practice. Testing on training data alone can produce overly optimistic results because the agent has already learned from those experiences.

---

## 9. Why are multiple training runs important?

PPO training is stochastic due to random initialization, exploration, and optimization. Different runs can converge to different policies, leading to different performance. Running multiple experiments provides a more reliable assessment than relying on a single result.

---

## 10. Main takeaway from this week's assignment

This assignment demonstrated how reinforcement learning can be applied to historical financial data while avoiding data leakage through chronological train-test splits. The PPO agent consistently outperformed a random policy but was unable to consistently beat the buy-and-hold baseline. The experiments also highlighted the variability of reinforcement learning training and the importance of evaluating on unseen future market data.