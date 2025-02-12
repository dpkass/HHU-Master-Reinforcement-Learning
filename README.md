# HHU Master Reinforcement Learning

This repository contains exercises and implementations from the **Reinforcement Learning** course at Heinrich Heine University (HHU), taught by **Prof. Milica Gašić**. The course follows a structured approach, covering fundamental and advanced reinforcement learning concepts.

## 📌 Course Overview

The exercises in this repository cover a range of reinforcement learning topics, from **Dynamic Programming** to **Deep Reinforcement Learning** methods. Each exercise consists of a **PDF assignment**, **Jupyter notebooks**, and necessary Python scripts.

---

## 📂 Repository Structure

```
HHU-Master-Reinforcement-Learning
│── Exercise 1   → Three-state MDP, transition probabilities
│── Exercise 2   → Discounted returns, policy iteration
│── Exercise 3   → Recursive Bellman equations, value iteration
│── Exercise 4   → Monte Carlo prediction & control
│── Exercise 5   → TD(0) vs. Monte Carlo prediction
│── Exercise 6   → SARSA and Q-learning (Cliff Walking)
│── Exercise 7   → Expected SARSA, Double Q-learning, n-step TD
│── Exercise 8   → λ-returns and function approximation
│── Exercise 9   → Multi-step Double DQN, TD(λ)
│── Exercise 10  → REINFORCE Algorithm (Policy Gradient)
│── Exercise 11  → Advantage Actor-Critic (A2C)
│── Exercise 12  → Proximal Policy Optimization (PPO)
│── Exercise 13  → Batch-Constrained Q-learning (BCQ)
└── requirements.txt → List of required dependencies
```

---

## ⚙️ Installation & Setup

To set up the environment, install the required dependencies:

```sh
pip install -r requirements.txt
```

---

## 📚 Exercises Details

### Exercise 1: Three-State MDP
- **Concepts**: Markov Decision Process (MDP), transition probabilities.
- **Files**: `three-state-mdp.ipynb`, `exercise1.pdf`

### Exercise 2: Discounted Returns & Policy Iteration
- **Concepts**: Discounting, Bellman equations, policy iteration.
- **Files**: `policy-iteration.ipynb`, `exercise2.pdf`

### Exercise 3: Bellman Equations & Value Iteration
- **Concepts**: Bellman equations, action-value functions, value iteration.
- **Files**: `value-iteration.ipynb`, `exercise3.pdf`

### Exercise 4: Monte Carlo Methods
- **Concepts**: Monte Carlo prediction & control.
- **Files**: `monte-carlo-prediction.ipynb`, `monte-carlo-control.ipynb`, `exercise4.pdf`

### Exercise 5: Temporal Difference (TD) Learning
- **Concepts**: TD(0), Monte Carlo, batch TD.
- **Files**: `mc-td-prediction.ipynb`, `exercise5.pdf`

### Exercise 6: SARSA & Q-Learning
- **Concepts**: On-policy & off-policy learning.
- **Files**: `td-control.ipynb`, `exercise6.pdf`

### Exercise 7: Expected SARSA & Double Q-Learning
- **Concepts**: Expected SARSA, Double Q-learning, n-step returns.
- **Files**: `double-q-learning.ipynb`, `td-control-2.ipynb`, `td-returns.ipynb`, `exercise7.pdf`

### Exercise 8: λ-Return & Linear Prediction
- **Concepts**: λ-returns, function approximation.
- **Files**: `lambda-returns.ipynb`, `linear-prediction.ipynb`, `exercise8.pdf`

### Exercise 9: Multi-Step DQN & TD(λ)
- **Concepts**: Multi-step Q-learning, TD(λ).
- **Files**: `linear-prediction-td-lambda.ipynb`, `multi-step-ddqn.ipynb`, `exercise9.pdf`

### Exercise 10: REINFORCE Algorithm
- **Concepts**: Policy gradient, softmax policies.
- **Files**: `reinforce.ipynb`, `exercise10.pdf`

### Exercise 11: Advantage Actor-Critic (A2C)
- **Concepts**: Advantage Actor-Critic (A2C), policy optimization.
- **Tasks**:
  - Implement **Advantage Actor-Critic (A2C)**.
  - Apply it to the **LunarLander** environment.
- **Files**: `actor-critic_v2.ipynb`, `exercise11.pdf`

### Exercise 12: Proximal Policy Optimization (PPO)
- **Concepts**: Policy optimization, PPO algorithm.
- **Tasks**:
  - Implement **Proximal Policy Optimization (PPO)**.
  - Apply it to the **LunarLander** environment.
- **Files**: `ppo.ipynb`, `exercise12.pdf`

### Exercise 13: Batch-Constrained Q-Learning (BCQ)
- **Concepts**: Offline RL, batch Q-learning.
- **Files**: `BCQ.ipynb`

---

## 📖 References

- **Sutton & Barto** – Reinforcement Learning: An Introduction.
- **OpenAI Gymnasium** – [gymnasium.farama.org](https://gymnasium.farama.org/).
- **Deep RL Resources** – Various research papers and online resources.
