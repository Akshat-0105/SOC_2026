# Week 4 Notes

## 1. Describe the reward curve from Task 1. Does the moving average trend upward over 100000 timesteps, or does it stay flat near zero? Was this what you expected? Why or why not?

The reward curve fluctuated throughout training and the moving average stayed close to zero instead of showing a steady upward trend. My final 20-episode average reward was approximately **0.12**. This was expected because the trading environment is based on a random walk, where future price movements are independent of past movements. Since there is no predictable pattern to exploit, PPO cannot consistently improve its rewards.

---

## 2. Compare the PPO agent's final 20-episode average reward to the random agent's average from Week 3 Task 4. Is there a meaningful difference? What does this tell you about whether PPO found a useful policy?

The PPO agent achieved a final 20-episode average reward of **0.12**, while the random agent from Week 3 produced rewards that also stayed close to zero. There is no meaningful difference between them. This suggests that PPO could not discover a significantly better policy because the environment does not contain exploitable information.

---

## 3. The transaction cost discourages the agent from changing position every step. Describe in your own words what type of policy would minimise transaction costs while still earning reward. Is that a good trading strategy?

A policy that minimizes transaction costs would avoid changing positions frequently and would only switch when it expects the expected profit to be greater than the trading cost. This reduces unnecessary losses due to frequent trading. In real financial markets, avoiding excessive trading is generally a good idea because every trade incurs costs.

---

## 4. You added history_len = 5 past returns to the state. On a true random walk, past returns contain no information about future returns. Given that, why might it still be worth including them in the state?

Although past returns do not predict future returns in a random walk, including a history window makes the environment more realistic and prepares it for future extensions. If the environment is later modified to include trends, momentum, or other market patterns, the agent will already have access to the necessary information. Therefore, the design is more flexible and scalable.

---

## 5. Look at your plots/cumulative_pnl.png. Do the three strategies (PPO, random, buy-and-hold) end up near the same cumulative P&L, or does one clearly outperform the others? Explain what this tells you about the environment.

With a transaction cost of **0.1**, my results were:

- PPO trained: **3.63**
- Random agent: **-6.71**
- Buy and Hold: **-1.32**

The PPO agent performed better than the two baseline strategies in this particular run. However, since the environment uses randomly generated price movements, these differences are likely due to randomness rather than a consistently better trading policy. Multiple training runs would be needed to verify whether the improvement is reliable.

---

## 6. When you removed the transaction cost (transaction_cost = 0.0), what happened to the three curves? Did the PPO agent perform better, worse, or the same relative to the random baseline?

With transaction cost removed, my results became:

- PPO trained: **-0.84**
- Random agent: **0.10**
- Buy and Hold: **-1.33**

In this run, the PPO agent performed worse than the random agent. This demonstrates that because the environment is a random walk, removing transaction costs does not guarantee better learning. The results still vary due to randomness, and no strategy consistently outperforms the others.

---

## 7. In your own words, explain why deterministic=True in model.predict during evaluation is different from deterministic=False during training. When would you want each?

During training, stochastic action selection allows the agent to explore different actions and collect diverse experiences. Exploration is necessary for learning an effective policy. During evaluation, deterministic=True makes the agent always choose the action with the highest learned probability so that its performance can be measured consistently without random exploration affecting the results.

---

## 8. After four weeks of RL (REINFORCE → PPO → trading environment → evaluation), what do you think is the hardest part of applying RL to financial markets? Is it the algorithm, the environment design, the reward, or something else?

I believe the hardest part is designing a realistic environment and reward function. Reinforcement learning algorithms such as PPO can learn effectively only if the environment accurately represents real-world trading conditions. Real financial markets contain trends, changing market regimes, transaction costs, slippage, and many external factors that are difficult to simulate. A poorly designed environment may cause an agent to learn strategies that perform well only in simulation but fail in real markets.

---

## 9. List 3 things you would want to add to TradingEnvV2 before you would trust an agent trained on it to trade real money. For each, say whether it changes the state, the action space, the reward, or the episode structure.

1. **Technical indicators (moving averages, RSI, MACD)**  
   - Changes: **State**

2. **Variable position sizing instead of only long, hold, and short**  
   - Changes: **Action Space**

3. **Slippage, realistic commissions, and risk penalties**  
   - Changes: **Reward**

---

## 10. Write down 3 questions about trading environments or RL for finance you want to discuss with your mentor.

1. How can a trading environment be made more realistic without making learning too difficult?

2. How should reward functions be designed so that the agent balances profitability with risk management?

3. What techniques are commonly used to prevent reinforcement learning agents from overfitting to historical financial data?