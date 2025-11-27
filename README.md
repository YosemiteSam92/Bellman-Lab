# Bellman-Lab: Vectorized Dynamic Programming for RL

A fully vectorized, Numpy-based implementation of Dynamic Programming algorithms (Value Iteration and Policy Iteration) to solve the `Frozen Lake` environment from Gymnasium.

## 🎯 Project Goal

The primary objective of this project is to bridge the gap between the mathematical formulation of Bellman equations and efficient, vectorized code.

While many introductory RL tutorials implement Dynamic Programming using nested for loops (iterating over every state and action sequentially), this project focuses on **vectorization**—using Linear Algebra to process the entire state-space simultaneously. This approach mirrors the mathematical notation found in R&D literature and significantly improves performance.

## 🧊 The Environment: Frozen Lake

We utilize the `FrozenLake-v1` environment from the **Gymnasium** library (the maintained fork of OpenAI Gym).

### Scenario
* The agent controls a character on a grid of frozen ice.
* Some tiles are safe (Frozen), others are dangerous (Holes), and one is the destination (Goal).
* The ice is slippery, meaning the agent's movement direction is stochastic.

### Technical Specifications

| Property | Description |
| :--- | :--- |
| **Grid Sizes** | Standard: 4x4 (16 states). Supports 8x8 or custom maps. |
| **Observation Space** | `Discrete(16)` (for 4x4). An integer from 0 to 15 representing the current tile index. |
| **Action Space** | `Discrete(4)`. 0: Left, 1: Down, 2: Right, 3: Up. |
| **Rewards** | +1 for reaching the Goal (G). 0 for all other transitions (including falling in Holes). |
| **Termination** | Falling into a Hole (H) or reaching the Goal (G). |

### Transition Dynamics (`env.P`)

In "God Mode" (Dynamic Programming), we have access to the environment's internal transition probabilities. In Gymnasium, this is exposed via the `env.P` attribute, which uses a nested dictionary structure:

env.P[state][action] = [(probability, next_state, reward, terminated), ...]

One of the first challenges of this project is converting this dictionary-based structure into 3D Numpy Tensors suitable for matrix multiplication:

* **Transition Tensor ($P$):** Shape `(S, A, S)`
* **Reward Tensor ($R$):** Shape `(S, A, S)`

### Stochasticity (`is_slippery`)

The environment includes an `is_slippery` parameter:

* **True (Default):** Movement is stochastic. Due to the slippery ice, the agent moves in the intended direction with probability $1/3$, and in perpendicular directions with probability $1/3$ each.
* **False:** Deterministic movement (useful for debugging and sanity checks).

## 🧮 Theoretical Foundation

We solve the Control Problem using the **Bellman Optimality Equation**:

$$
V^*(s) = \max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma V^*(s')]
$$

By treating the value function $V$ as a vector and the transitions $P$ as a tensor, we compute updates using `numpy.einsum` and broadcasting, eliminating Python-level loops over states.

## 🚀 Roadmap

1.  **Tensor-ification:** Bridge `env.P` to Numpy Tensors.
2.  **Vectorized Policy Evaluation:** Implement $V^\pi$ estimation.
3.  **Policy Iteration:** Full control loop.
4.  **Value Iteration:** Aggressive value updates.
5.  **Visualization:** Heatmap overlays and agent simulation.

