# Week 2 Notes

## 1. Describe the trend in episode returns.

The episode returns fluctuated throughout training. There were occasional improvements around the middle of training, but the learning was not stable and the moving average later decreased instead of consistently increasing. This indicates that the policy learned some useful behavior temporarily but was unable to maintain it.

---

## 2. Compare the trained policy to a random policy.

The trained policy was able to balance the pole for longer than a random policy during some episodes, especially in the middle of training. However, its performance was inconsistent and it eventually became similar to a random policy in later episodes. A random policy makes completely uninformed decisions, while the trained policy attempts to use the learned probabilities to choose actions.

---

## 3. Try at least one hyperparameter change.

I experimented with the learning process by modifying the implementation and observing the training behavior. Although the training dynamics changed slightly, the policy still showed unstable learning. This demonstrates that REINFORCE is sensitive to implementation details and hyperparameter choices.

---

## 4. What does the REINFORCE loss −logπ(at|st)Gt try to do?

The REINFORCE loss increases the probability of actions that lead to high future rewards while decreasing the probability of actions that produce poor outcomes. The negative sign is used because PyTorch minimizes a loss function, whereas reinforcement learning aims to maximize the expected cumulative reward.