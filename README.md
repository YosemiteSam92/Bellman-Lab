# Bellman-Lab: Vectorized Dynamic Programming for RL

![Agent Simulation](assets/agent_run.gif)

A fully vectorized, Numpy-based implementation of Dynamic Programming algorithms (Value Iteration and Policy Iteration) to solve the `Frozen Lake` environment from Gymnasium.

## Project Goal

The primary objective of this project is to bridge the gap between the mathematical formulation of Bellman equations and efficient, vectorized code.

While many introductory RL tutorials implement Dynamic Programming using nested for loops (iterating over every state and action sequentially), this project focuses on **vectorization**—using Linear Algebra to process the entire state-space simultaneously. This approach mirrors the mathematical notation found in R&D literature and significantly improves performance.

## The Environment: Frozen Lake

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

`env.P[state][action] = [(probability, next_state, reward, terminated), ...]`

One of the first challenges of this project is converting this dictionary-based structure into 3D Numpy Tensors suitable for matrix multiplication:

* **Transition Tensor ($P$):** Shape `(S, A, S)`
* **Reward Tensor ($R$):** Shape `(S, A, S)`

### Stochasticity (`is_slippery`)

The environment includes an `is_slippery` parameter:

* **True (Default):** Movement is stochastic. Due to the slippery ice, the agent moves in the intended direction with probability $1/3$, and in perpendicular directions with probability $1/3$ each.
* **False:** Deterministic movement (useful for debugging and sanity checks).

## Theoretical Foundation

We solve the **Control Problem** (finding the optimal policy $\pi^*$) using the Bellman equations. This process is broken down into two interacting components:

1.  **Prediction (Policy Evaluation):** Calculating the value function $V^\pi$ for a specific policy.
2.  **Control (Policy Improvement):** using $V^\pi$ to find a better policy $\pi'$.

By treating the value function $V$ as a vector and the transitions $P$ as a tensor, we compute updates using `numpy.einsum` and broadcasting, eliminating Python-level loops over states.

## Vectorized Policy Evaluation

Instead of iterating through states and actions with slow Python loops, we evaluate a policy $\pi$ by reducing the 3D dynamics tensors into 2D/1D matrices specific to that policy.

We use `numpy.einsum` to perform these contractions efficiently:

### 1. Policy Transition Matrix ($P^\pi$)
* **Shape:** `(S, S)`
* **Concept:** If I am in state $s$ and follow policy $\pi$, what is the probability I end up in state $s'$?
* **Math:** $P^\pi_{ss'} = \sum_{a} \pi(a|s) P(s'|s,a)$
* **Code:** `P_pi = np.einsum('sa, sak -> sk', policy, P)`

### 2. Expected Reward Vector ($r^\pi$)
* **Shape:** `(S,)`
* **Concept:** If I am in state $s$ and follow policy $\pi$, how much immediate reward do I expect on average?
* **Math:** $r^\pi_s = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s,a) R(s,a,s')$
* **Code:** `r_pi = np.einsum('sa, sak, sak -> s', policy, P, R)`

### 3. The Update Rule
With these pre-computed matrices, the Bellman Expectation Equation becomes a clean linear algebra operation. We iterate until convergence:

$$V_{new} = r^\pi + \gamma (P^\pi \cdot V_{old})$$

    V_new = r_pi + gamma * (P_pi @ V)

#### Note on Analytical Solution
While we use iterative policy evaluation to demonstrate the underlying dynamic programming mechanism, the Bellman Expectation Equation for a fixed policy is a linear system:

$$(I - \gamma P^\pi)V^\pi = r^\pi$$

For small state spaces like Frozen Lake ($N=16$), we could technically solve for $V^\pi$ directly by inverting the matrix:

$$V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$$

However, matrix inversion carries a computational complexity of $O(N^3)$. We deliberately choose the iterative approach ($O(N^2)$ per iteration) because it is more numerically stable for large matrices and scales better to high-dimensional state spaces, mirroring the logic used in deep reinforcement learning.

## Vectorized Policy Improvement

Once we have the value of the current policy ($V^\pi$), we can improve it by acting greedily. This requires calculating the Action-Value function $Q(s, a)$.

$$Q(s, a) = \sum_{s'} P(s'|s,a) [ R(s,a,s') + \gamma V(s') ]$$

We vectorize this calculation by splitting it into two parts:

### 1. Expected Immediate Reward
The reward $R$ is weighted by the transition probabilities $P$.
* **Code:** `expected_immediate_reward = np.einsum('sak, sak -> sa', P, R)`

### 2. Expected Future Value
The discounted value of the next state, weighted by transition probabilities.
* **Code:** `expected_future_value = np.einsum('sak, k -> sa', P, V)`

### 3. The Greedy Update
We combine these to get the Q-values and simply select the best action for each state (row).

    Q = expected_immediate_reward + gamma * expected_future_value
    best_actions = np.argmax(Q, axis=1)

## Policy Iteration

The full **Policy Iteration** algorithm simply loops between the two vector operations defined above:

1.  **Evaluate:** Get $V^\pi$ for the current policy using the linear algebra update.
2.  **Improve:** Update $\pi$ to be greedy with respect to $V^\pi$.
3.  **Repeat:** Continue until the policy stops changing (Stability).

This method is guaranteed to converge to the optimal policy $\pi^*$.

## Value Iteration

While Policy Iteration alternates between two distinct phases (evaluation and improvement), **Value Iteration** combines them into a single, aggressive update step.

Instead of evaluating a specific policy until convergence, we update the value of a state directly by assuming we always take the best action available:

$$V_{new}(s) = \max_a Q(s, a)$$

We vectorize this efficiently:

    # 1. Calculate Q-values for all actions (S, A)
    Q = self.get_q_values(V, gamma)

    # 2. Update V to be the max Q-value (S,)
    V_new = np.max(Q, axis=1)

By repeating this update until $V$ stops changing, we converge to $V^*$. This approach is often computationally cheaper per iteration than Policy Iteration because it avoids solving a linear system or running a nested evaluation loop.

## Results & Analysis

After converging to the optimal Value Function $V^*$, we project the values back onto the 4x4 grid.

* **Color Scale:** Lighter (Yellow) tiles represent higher value states (closer to Goal). Darker (Purple) tiles represent low value states (near Holes).
* **Arrows:** The arrows represent the optimal policy $\pi^*(s)$.

![Optimal Value Heatmap](assets/value_heatmap.png)

### The "Wall Bumping" Strategy

An interesting emergent behavior is seen at the **Start State (0,0)** (top-left). The policy points **LEFT**, directly into the wall.

At first glance, this seems wrong. The goal is down-right, so why not go **DOWN**? The answer lies in the stochastic mechanics of the environment.

* **Slippery Rule:** If you choose an action, you have a $1/3$ chance of going that way, and a $1/3$ chance of slipping to either perpendicular side.
* **Wall Rule:** If you move into a wall, you bounce back and stay in the same state.

Let's look at the math derived from our Value Function $V^*$:
* $V(\text{Start}) \approx 0.542$
* $V(\text{Down}) \approx 0.558$ (Higher Value!)
* $V(\text{Right}) \approx 0.499$ (Lower Value - dangerous path)

#### Scenario A: The "Intuitive" Move (DOWN)
If the agent chooses **DOWN**, it risks slipping **RIGHT** (towards the danger zone and a lower-value tile).
* $33\%$ Down $\rightarrow$ lands on $V=0.558$ (Success)
* $33\%$ Left $\rightarrow$ hits wall, stays on $V=0.542$ (Neutral)
* $33\%$ Right $\rightarrow$ slips to $V=0.499$ (**Failure**, lower state value than the current one)

$$E[V] \approx 0.33(0.558) + 0.33(0.542) + 0.33(0.499) \approx \mathbf{0.533}$$

#### Scenario B: The "Wall Bump" (LEFT)
If the agent chooses **LEFT**, it drives into the wall. This removes the "Right" slip entirely.
* $33\%$ Left $\rightarrow$ hits wall, stays on $V=0.542$ (Neutral)
* $33\%$ Up $\rightarrow$ hits wall, stays on $V=0.542$ (Neutral)
* $33\%$ Down $\rightarrow$ slips to $V=0.558$ (Success, higher state value than the current one)

$$E[V] \approx 0.33(0.542) + 0.33(0.542) + 0.33(0.558) \approx \mathbf{0.547}$$

**Conclusion:** Since $0.547 > 0.533$, the optimal move is to grind against the wall and wait for the ice to slip you Down, rather than risking a slip to the Right. The agent has learned to exploit the environment's physics to mitigate risk.

## Roadmap

1.  **Tensor-ification:** Bridge `env.P` to Numpy Tensors. [Complete]
2.  **Vectorized Policy Evaluation:** Implement $V^\pi$ estimation. [Complete]
3.  **Policy Iteration:** Full control loop. [Complete]
4.  **Value Iteration:** Aggressive value updates. [Complete]
5.  **Visualization:** Heatmap overlays and agent simulation. [Complete]