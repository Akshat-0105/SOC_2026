# Week 1 - Assignment 1 Notes

# Part 1 – Random Agent

## Random Agent Results

| Run | Reward |
|------|--------|
| 1 | 20 |
| 2 | 13 |
| 3 | 17 |
| 4 | 18 |
| 5 | 27 |

**Average Reward:** 19.0

### 1. What does each of the 4 numbers in the observation mean?

The observation consists of four values:

- Cart Position – Horizontal position of the cart.
- Cart Velocity – Speed and direction of the cart.
- Pole Angle – Angle of the pole relative to the vertical.
- Pole Angular Velocity – Speed at which the pole is rotating.

---

### 2. Why does the pole fall so quickly?

The agent selects actions randomly without considering the state of the environment. Since there is no strategy to balance the pole, it quickly falls.

---

### 3. The reward is +1 per step the pole stays up. What's the maximum total reward possible?

The maximum reward for CartPole-v1 is **500**.

---

# Part 2 – Hand-Coded Policy

## Policy Used

```python
def my_policy(observation):
    cart_pos, cart_vel, pole_angle, pole_ang_vel = observation

    if pole_angle > 0:
        return 1
    else:
        return 0
```

## Episode Scores

| Episode | Reward |
|----------|--------|
| 1 | 37 |
| 2 | 51 |
| 3 | 51 |
| 4 | 47 |
| 5 | 45 |

**Average Reward:** 46.2

---

### 4. Write down the rule you used.

If the pole angle is positive, push the cart to the right. Otherwise, push it to the left.

---

### 5. What was your average score?

The average score over five episodes was **46.2**.

---

### 6. Did the rule work every time, or did it sometimes fail? Why?

The rule improved the performance compared to random actions, but it still failed sometimes because it only considers the pole angle and ignores the cart position, cart velocity, and pole angular velocity.

---

### 7. Can you think of a smarter rule?

A smarter rule would use all four state variables instead of only the pole angle. It could also consider how quickly the pole is falling and the cart's movement before choosing an action.

---

# Part 3 – MountainCar-v0

### 8. Which environment did you try?

MountainCar-v0.

---

### 9. What is the agent trying to do?

The agent's goal is to drive the car to the top of the mountain on the right side.

---

### 10. What does a state look like? What are the possible actions?

The state consists of two values:

- Car position
- Car velocity

Possible actions:

- 0 – Push Left
- 1 – No Push
- 2 – Push Right

---

### 11. Does random play ever solve it? Why or why not?

Random play almost never solves the environment because the car does not have enough power to climb the hill directly. It needs to build momentum by moving back and forth before attempting to reach the goal.

---

# Part 4 – Big Picture Reflection

### 12. In your own words, what is the agent's goal in any RL problem?

The goal of an RL agent is to interact with its environment and choose actions that maximize the total reward over time.

---

### 13. Why is random not a good strategy, and what would the agent need to do instead?

Random actions ignore the current state of the environment and usually lead to poor performance. Instead, the agent should learn which actions are best in different situations based on past experience.

---

### 14. What does it mean for an agent to "learn"?

Learning means improving the policy through experience. As the agent interacts with the environment, it changes its behavior so that it performs better in future episodes.

---

### 15. Three questions I would like to ask my mentor

1. How does an RL agent discover a good policy without being explicitly programmed?
2. How many episodes are typically required before an RL agent learns a useful strategy?
3. Why are neural networks commonly used in modern reinforcement learning algorithms?