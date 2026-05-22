# Week 1: What is Reinforcement Learning?

Welcome! This week is about **understanding what reinforcement learning is** before we write a single line of algorithm code. No PPO, no policy gradients, no math derivations yet — just intuition and curiosity.

## 🎯 Goal of the Week

By the end of this week you should be able to answer, in your own words:

- What is an **agent**, an **environment**, a **state**, an **action**, and a **reward**?
- How is RL different from supervised learning?
- What does it mean for an agent to "learn from experience"?
- What kinds of problems is RL good for?

That's it. No more, no less.

## 🧭 What This Week Is NOT

- ❌ Implementing PPO, SAC, MADDPG, or any other algorithm
- ❌ Reading research papers
- ❌ Heavy math (probability, gradients, Bellman equations)
- ❌ Training a "good" agent

All of that comes in later weeks. If it feels like we're going slow — good. That's the point.

## 📅 Suggested Pace (5–6 days, ~2–3 hours/day)

### Day 1 — Read & Watch (2 hours)
Read [resources/what_is_rl.md](resources/what_is_rl.md). It's a friendly, no-math walkthrough of the core ideas.

Then watch **one** of these (whichever style you like):
- 📺 [Reinforcement Learning in 3 Hours (freeCodeCamp)](https://www.youtube.com/watch?v=Mut_u40Sqz4) — watch the first 20 minutes
- 📺 [What is Reinforcement Learning? (DeepMind, 5 min)](https://www.youtube.com/watch?v=2pWv7GOvuf0) — first 15 minutes of David Silver Lecture 1

### Day 2 — Setup (1–2 hours)
Get your environment working. Follow [setup/setup_guide.md](../setup/setup_guide.md), then run:

```bash
python setup/verify_setup.py
```

If you see ✅ for Python, PyTorch, Gymnasium, NumPy — you're good. **Don't worry about anything else yet.**

### Day 3–4 — Hands-on Assignment (3–4 hours)
Do [assignments/assignment1_explore.md](assignments/assignment1_explore.md). You'll:
1. Watch a random agent fail at CartPole
2. Write a simple hand-coded rule that does better
3. Reflect in writing on what you saw

**No neural networks. No training. Just observe and tinker.**

### Day 5 — Reflect & Explore (1–2 hours)
- Skim the [FAQ](../shared/FAQ.md) — don't try to understand everything, just see what's there
- Try one more Gymnasium environment (`MountainCar-v0` or `Acrobot-v1`) with random actions, just to see different problems
- Write down 3 questions you have about RL — bring them to your mentor

### Day 6 (Optional) — Go Deeper
If you're hungry for more, browse [week1_advanced/](../week1_advanced/) — that's the "implement PPO from scratch" track. It's there when you're ready.

## 📚 Resources (only what you need)

### Just read this one thing
- [resources/what_is_rl.md](resources/what_is_rl.md) — the gentle intro

### Pick ONE video (don't watch all)
- [David Silver — Lecture 1, first 15 min](https://www.youtube.com/watch?v=2pWv7GOvuf0) — academic style
- [freeCodeCamp — RL in 3 hours, first 20 min](https://www.youtube.com/watch?v=Mut_u40Sqz4) — practical style
- [Two Minute Papers — AlphaGo](https://www.youtube.com/watch?v=8tq1C8spV_g) — inspiring style

### Optional reading (if curious)
- [Spinning Up — A Short Introduction to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — slightly more technical
- [HuggingFace Deep RL Course — Unit 1](https://huggingface.co/learn/deep-rl-course/unit1/introduction) — interactive

## ✅ Done When

You can check off:

- [ ] Environment is set up (`verify_setup.py` passes)
- [ ] You read [what_is_rl.md](resources/what_is_rl.md)
- [ ] You watched at least one intro video
- [ ] You finished [assignment 1](assignments/assignment1_explore.md)
- [ ] You wrote down 3 questions for your mentor

## 💬 Talk to Your Mentor

After Week 1, you should have a conversation with your mentor about:
- What surprised you about RL?
- What concept still feels fuzzy?
- Which of the three video styles clicked best for you?

That conversation decides how Week 2 is paced for **you specifically**.

## ⏭️ Looking Ahead

In Week 2 we'll start writing actual RL code — beginning with the simplest possible learning algorithm (REINFORCE). For now: **just understand the problem before we try to solve it.**

---

**Don't rush.** This week is the foundation. If you skip it, everything later will feel confusing.

Ready? Start with [resources/what_is_rl.md](resources/what_is_rl.md).
