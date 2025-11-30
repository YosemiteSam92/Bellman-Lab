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


def plot_heatmap(env: gym.Env, V: np.ndarray, policy: np.ndarray, filename: str="value_heatmap.png") -> None:
    """
    Visualizes the state-value function V as a heatmap 
    and overlays the optimal policy as arrows.
    """

# 1. Reshape V to match the grid (4x4)
    desc = env.unwrapped.desc.astype(str) # The grid map chars (S,F,H,G)
    rows, cols = desc.shape
    V_grid = V.reshape(rows, cols)

    # 2. Plotting Setup
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Render the heatmap
    im = ax.imshow(V_grid, cmap='viridis')
    
    # Add a colorbar
    plt.colorbar(im, label='State Value (Expected Discounted Reward)')

    # 3. Overlay Text and Arrows
    arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    for i in range(rows):
        for j in range(cols):
            
            state_index = i * cols + j
            tile_type = desc[i, j]
            value = V_grid[i, j]
            best_action = np.argmax(policy[state_index])
            arrow = arrows[best_action]

            # --- DYNAMIC TEXT LOGIC ---
            
            # 1. Determine Text Content
            if i == 0 and j == 0:
                text = f"START\n{arrow}\n({value:.3f})"
            elif tile_type == 'H':
                text = "HOLE\n(0.00)"
            elif tile_type == 'G':
                text = "GOAL\n(1.00)"
            else:
                text = f"{arrow}\n{value:.3f}"

            # 2. Determine Text Color for Contrast
            # Viridis colormap: Low values are dark (purple), High values are light (yellow)
            # We use white text for dark backgrounds (value < 0.5)
            # We use black text for light backgrounds (value > 0.5)
            text_color = "white" if value < 0.5 else "black"
            
            # Special override for Hole/Goal if needed, but the value-based logic is usually best.
            # (Goal is 1.0 -> Yellow -> Black text is most readable)
            
            ax.text(j, i, text, ha='center', va='center', color=text_color, fontweight='bold')

    # Formatting
    ax.set_title("Optimal Value Function & Policy (Slippery Ice)", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Heatmap saved to {filename}")
    plt.close()


def record_agent_run(env: gym.Env, policy: np.ndarray, filename: str="agent_run.gif") -> None:
    """
    Runs a simulation of the agent following the provided policy
    and saves the frames as a GIF.
    """
    frames = []
    
    # 1. Reset Environment
    state, _ = env.reset()
    
    # Capture the starting frame
    frames.append(env.render())

    done = False
    step_count = 0
    max_steps = 100 # Safety limit to prevent infinite loops if agent gets stuck

    print("Simulating agent...")
    
    while not done and step_count < max_steps:
        # 2. Select Action
        # The policy array is (S, A), we take the best action for the current state
        action = np.argmax(policy[state])
        
        # 3. Step
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 4. Capture Frame
        frames.append(env.render())
        step_count += 1

    env.close()

    # 5. Save GIF
    # duration is time per frame in milliseconds
    imageio.mimsave(filename, frames, fps=2) 
    print(f"Simulation GIF saved to {filename} ({step_count} steps)")

if __name__ == "__main__":

    env, optimal_policy, optimal_V = run_solver()

    plot_heatmap(env, optimal_V, optimal_policy)

    record_agent_run(env, optimal_policy)