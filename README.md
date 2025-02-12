# Reinforcement Learning

**Author:** Taha El Amine Kassabi\
**Course:** Reinforcement Learning (WS 2024/25)\
**Instructor:** Prof. Dr. Milica Gašić\
**University:** Heinrich Heine University Düsseldorf (HHU)

---

## 📚 Overview

This repository contains my solutions to **13 programming exercises** from the *Reinforcement Learning* course at HHU.\
Each week builds on the previous one, covering **value-based**, **policy-based**, and **deep reinforcement learning** methods.\
All implementations are in Python, using **NumPy**, **Matplotlib**, and **PyTorch**, where applicable.

---

## 📂 Repository Structure

```
Exercises/         # Basically the same as Topics Covered
requirements.txt → Package list
```

---

## 🧠 Topics Covered

| Week | Algorithms & Concepts                             |
|------|---------------------------------------------------|
| 01   | MDPs, transition matrices, value functions        |
| 02   | Discounting, Bellman equations, policy iteration  |
| 03   | Recursive Bellman, value iteration                |
| 04   | Monte Carlo prediction & control                  |
| 05   | TD(0) vs. Monte Carlo, bootstrapping              |
| 06   | SARSA, Q-learning, on- vs. off-policy learning    |
| 07   | Expected SARSA, Double Q-learning, n-step returns |
| 08   | λ-return updates, linear approximation, SGD       |
| 09   | TD(λ), Multi-step Double DQN                      |
| 10   | REINFORCE, softmax policies, stochastic updates   |
| 11   | Actor-Critic (A2C), value baseline, LunarLander   |
| 12   | PPO, clipped surrogate loss, trajectory sampling  |
| 13   | BCQ, offline reinforcement learning               |

---

## 📁 Setup

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

Run each notebook in the corresponding exercise folder using Jupyter or VSCode.

```bash
jupyter lab td-control.ipynb
```

---

## 📊 Notes

- Environments follow OpenAI Gym / Gymnasium APIs.
- Some models trained on **CliffWalking-v0** and **LunarLander-v2**.
- All algorithms written from scratch.

---

## 🖊️ References

- Sutton & Barto — *Reinforcement Learning: An Introduction*
- OpenAI Gymnasium — [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
- Assorted papers and slides provided during the course

