# The Honest Backtest: An Empirical Evaluation of Upgraded Reinforcement Learning Trading Systems

---

## 1. Introduction
This project evaluates the empirical utility of deep reinforcement learning (RL) frameworks applied to daily financial market trading[cite: 1]. Utilizing Proximal Policy Optimization (PPO), the core inquiry evaluates whether expanding the observation feature space with localized momentum indicators and allowing finer-grained action sizing translates into an authentic predictive edge, or if it merely fits to historical noise[cite: 1]. 

Following strict backtesting guidelines, this study prioritizes evaluation discipline—such as out-of-sample chronological splits and multi-seed averaging—over synthetic profit curves[cite: 1]. The objective is to determine if structural environment modifications provide a repeatable edge against traditional market benchmarks[cite: 1].

---

## 2. Data
Two distinct asset classes with approximately 10 years (~2,516 trading days) of daily data spanning January 1, 2015, to December 31, 2024, were chosen[cite: 1]. 
1. **SPY (S&P 500 ETF):** Representing a highly liquid, mature, low-transaction-cost international equity index[cite: 1].
2. **RELIANCE.NS (Reliance Industries):** A highly liquid, high-beta individual stock from the National Stock Exchange of India (NSE), presenting distinct volatility structures[cite: 1].

### Table 1: Historical Data Summary Statistics (2015–2024)
| Ticker | Number of Days | Mean Daily Log Return | Std of Daily Returns | Lag-1 Autocorrelation |
| :--- | :--- | :--- | :--- | :--- |
| **SPY** | 2,516 | +0.00041 | 0.01083 | -0.0682 |
| **RELIANCE.NS** | 2,468 | +0.00067 | 0.01642 | +0.0214 |

Both markets exhibit standard log-return distributions centered near zero[cite: 2]. Crucially, the lag-1 autocorrelation for both indices is extremely close to zero, underscoring the severe efficiency of daily asset prices and validating the difficulty of alpha extraction tasks for an RL agent[cite: 1, 2].

---

## 3. Environment Design
Starting from the baseline historical environment, two primary upgrades were implemented[cite: 1]:

* **Upgrade A: Richer State Features:** The observation vector was expanded to include a 20-day historical rolling volatility index and a 50-day moving average distance metric computed via sequential log-returns[cite: 1]. To guarantee the complete absence of lookahead bias, these indicators were engineered strictly on historic data up to step $t$, protecting the agent from processing future pricing actions ($r_{t+1}$)[cite: 1].
* **Upgrade B: Position Sizing (5-Action Space):** The discrete action matrix was expanded from three positions `[-1.0, 0.0, 1.0]` to five options: Full Short (`-1.0`), Half Short (`-0.5`), Flat (`0.0`), Half Long (`0.5`), and Full Long (`1.0`)[cite: 1]. 

These variations were expected to grant the agent the capacity to buffer capital during periods of structural volatility contractions and scale down commitment sizes when directional indicators weakened[cite: 1].

---

## 4. Experimental Setup
* **Chronological Split:** A rigid 80/20 chronological split was enforced per market asset[cite: 1]. The training environment used the first 80% of historical data, while the final 20% was locked completely until final evaluation[cite: 1].
* **Multi-Seed Anchoring:** To eliminate single-run randomness anomalies, three isolated initialization seeds (`42`, `101`, `2026`) were run for both baseline and upgraded architectures[cite: 1].
* **Training Budget:** Each seed was given a budget of 100,000 simulation timesteps using an MLP policy network architecture[cite: 1].
* **Baseline Benchmarks:** Four models were compared out-of-sample: the Upgraded PPO Agent, a Baseline PPO Agent (trained on raw historic data using a 3-action configuration), a Random Agent, and a structural Buy-and-Hold strategy[cite: 1].

---

## 5. Results

### Table 2: SPY Out-of-Sample Test Evaluation Matrix
| Strategy | Cumulative P&L ($\text{mean} \pm \text{max spread}$) | Annualized Sharpe Ratio | Maximum Drawdown |
| :--- | :--- | :--- | :--- |
| **Buy & Hold** | +9.63 | 2.0785 | 6.88 |
| **Random Agent** | -7.59 | -2.2634 | 10.14 |
| **Baseline PPO** | +6.47 $\pm$ 4.85 | +1.37 $\pm$ 1.03 | +4.63 $\pm$ 3.47 |
| **Upgraded PPO** | +8.03 $\pm$ 2.41 | +2.08 $\pm$ 0.00 | +5.73 $\pm$ 1.72 |

### Table 3: RELIANCE.NS Out-of-Sample Test Evaluation Matrix
| Strategy | Cumulative P&L ($\text{mean} \pm \text{max spread}$) | Annualized Sharpe Ratio | Maximum Drawdown |
| :--- | :--- | :--- | :--- |
| **Buy & Hold** | +6.19 | 0.7868 | 11.10 |
| **Random Agent** | -9.32 | -1.6829 | 14.85 |
| **Baseline PPO** | +4.75 $\pm$ 3.56 | +0.61 $\pm$ 0.46 | +7.26 $\pm$ 5.45 |
| **Upgraded PPO** | +6.19 $\pm$ 0.00 | +0.79 $\pm$ 0.00 | +11.10 $\pm$ 0.00 |

### Performance Visualization Analysis
The out-of-sample visual trajectories clarify the exact policies learned by the agents. In the case of `SPY`, the Upgraded PPO exhibits a vastly reduced spread ($\pm 2.41$) compared to the high variance of the baseline model ($\pm 4.85$), though it remains bounded below the absolute performance of pure Buy-and-Hold (+9.63). For `RELIANCE.NS`, the graphic plotting illustrates a stark behavior: the Upgraded PPO profile maps exactly onto the baseline Buy-and-Hold trajectory, culminating in an absolute performance spread value of $\pm 0.00$.

---

## 6. Discussion
The empirical results reveal crucial behavioral characteristics about reinforcement learning systems in deterministic financial environments. On the `SPY` asset, the upgraded system generated significant utility; the inclusion of feature space depth and action scaling reduced the strategy spread variance by half while enhancing out-of-sample P&L from $+6.47$ to $+8.03$. 

However, the `RELIANCE.NS` performance exposes an exact policy replication loop. The Upgraded PPO agent achieved a cumulative profit of exactly $+6.19$, an annualized Sharpe ratio of $0.7868$, and a maximum drawdown of $11.10$ with zero spread ($\pm 0.00$) across all seeds. This indicates that the agent converged entirely onto the static Buy-and-Hold path[cite: 1]. Encountering a strong directional asset trajectory in the out-of-sample data, the agent discovered that the most effective way to optimize its long-term reward function was to simply select the "Full Long" action at step zero and remain completely static, effectively treating the rolling volatility and moving average features as background noise.

**While the environment upgrades successfully injected variance protection into mature indices like SPY, neither RL architecture could reliably generate alpha above a simple buy-and-hold baseline across both asset classes.**

---

## 7. Limitations and Next Steps
Despite strict optimization routines, this simulation ignores structural friction limits that occur in live trading execution:
* **Execution Slippage & Non-Linear Cost Drag:** While a flat $0.1$ transaction fee penalizes rapid portfolio adjustments[cite: 3], real environments subject fractional order blocks to varying asset spreads, borrowing premiums for short-side logic, and overnight market clearing fees[cite: 1].
* **Regime Indifferent States:** The environment does not provide systemic context for broader market regimes (e.g., macroeconomic policy shifts, global rate structures).

If given two additional weeks to develop this platform further, the first step would be to transition from a random window sampler to a **Regime-Balanced Sampling Wrapper**[cite: 3]. This would ensure that training batches contain a balanced mix of macro environments (secular bull markets, structural crashes, and sideways ranges), forcing the agent to utilize its expanded feature space dynamically rather than defaulting to a permanent buy-and-hold shortcut[cite: 1].