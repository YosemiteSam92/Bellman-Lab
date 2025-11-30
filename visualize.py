import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
from typing import Tuple
from BellmanModel import BellmanModel

def run_solver() -> Tuple[gym.Env, np.ndarray, np.ndarray]:
    """
    Initializes the environment and the Bellman model,
    running value iteration to get the optimal policy and value function.

    Returns:
        Tuple[gym.Env, np.ndarray, np.ndarray]:
            - environment
            - optimal policy: Shape (S, A)
            - optimal value function: Shape (S,)
    """

    # 1. Setup the environment
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="rgb_array")

    # 2. Initialize model
    model = BellmanModel(env)

    # 3. Solve the environment using value iteration
    print("Solving the environment using Value Iteration...")
    optimal_policy, optimal_V = model.value_iteration(gamma=0.99)
    print("Solved!")

    return env, optimal_policy, optimal_V

if __name__ == "__main__":
    env, optimal_policy, optimal_V = run_solver()