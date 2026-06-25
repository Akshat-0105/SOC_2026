# Week 3 Notes

## 1. Which hyperparameter change hurt learning the most? What does this tell you about gradient-based optimisation?

Changing the discount factor to gamma = 0.50 (Run C) produced one of the lowest final average returns. A smaller gamma makes the agent focus more on immediate rewards and ignore long-term consequences. This shows that reinforcement learning performance is highly sensitive to hyperparameter selection.

---

## 2. REINFORCE updates after every single episode. Why does using one trajectory’s data make the gradient estimate noisy?

Each episode is only one sample of the environment, so the collected rewards can vary significantly due to randomness. As a result, the gradient estimate has high variance, making learning unstable and slower.

---

## 3. PPO uses a clipped probability ratio. In your own words, what problem does the clip prevent?

The clipping mechanism prevents the policy from changing too much after a single update. This keeps learning stable by avoiding excessively large policy updates that could reduce performance.

---

## 4. PPO trains on mini-batches from collected experience. Why might this produce a smoother reward curve than REINFORCE?

PPO updates the policy using many samples collected before each optimization step, which reduces the variance of the gradient estimate. Mini-batch updates also make learning more stable than updating from a single episode.

---

## 5. Describe the two reward curves. Which learned faster? Which was more consistent? Were there any surprises?

The REINFORCE reward curve was noisy and showed inconsistent improvement throughout training. PPO learned much faster and achieved a much higher reward with a smoother learning curve. The biggest surprise was how quickly PPO reached near-perfect performance compared to REINFORCE.

---

## 6. List five features you would add to the trading environment state.

1. Moving average of recent prices to capture market trends.
2. Trading volume to indicate market activity.
3. Volatility estimate to measure market uncertainty.
4. Previous price history to identify short-term patterns.
5. Current portfolio value to help the agent manage risk.

---

## 7. What happens if the agent always stays long? Is that a good strategy on a random walk?

If the agent always stays long, it profits only when prices increase and loses when prices decrease. Since the price follows a random walk with no long-term trend, always staying long is not a reliable strategy and should produce an average reward close to zero over many episodes.

---

## 8. What are three limitations of this toy environment?

1. Prices follow a simple random walk instead of realistic market dynamics.
2. There are no transaction costs or trading fees.
3. The environment models only one asset and ignores factors such as liquidity and market impact.

---

## 9. What was the mean reward of the random agent over 10 episodes? Was it positive, negative, or near zero?

The random agent achieved a mean reward of **−2.73** over 10 episodes. The reward was slightly negative but still close to zero, which is expected because the environment follows a random walk and the agent acts randomly.

---

## 10. Three questions to discuss with my mentor

1. How can variance reduction techniques improve REINFORCE training?
2. Why does PPO generally outperform vanilla policy gradient methods?
3. What additional state features are most important when designing a realistic trading environment?
