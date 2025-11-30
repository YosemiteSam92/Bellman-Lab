import numpy as np
import gymnasium as gym
from typing import Tuple

class BellmanModel:
    def __init__(self, env: gym.Env) -> None:
        """
        Initializes the model by extracting the transition dynamics
        from the Gymnasium environment and converting them to tensors.

        Args:
            env (gym.env): The Gymnasium environment (e.g., FrozenLake-v1).
        """

        # use .unwrapped to get the base non-wrapped environment.
        self.env = env.unwrapped

        # 1. Get the shape of the world
        # S = number of states (16 for a 4x4 grid)
        # A = number of actions (4: Left, Down, Right, Up)
        self.S = env.observation_space.n
        self.A = env.action_space.n

        # 2. Initialize our tensors with zeros
        # P: the probability of moving from s to s' given action a
        # shape: (states, actions, next_states)
        self.P = np.zeros((self.S, self.A, self.S))

        # R: the reward received when moving from s to s' given action a
        # shape: (states, actions, next_states)
        self.R = np.zeros((self.S, self.A, self.S))

        # 3. Trigger the conversion
        self.convert_dynamics()

    def convert_dynamics(self) -> None:
        """
        Iterates over the environment's transition dynamics (P dictionary)
        and populates the P and R tensors accordingly.

        Reminder: env.P[state][action] = list of tuples, where each tuple is of the form:
                  (probability, next_state, reward, terminated)
        """

        # loop over every state (0 to 15)
        for s in range(self.S):
            
            # loop over every action (0 to 3)
            for a in range(self.A):

                # retrieve the list of possible transitions for this (s, a) pair
                transitions = self.env.P[s][a]

                # iterate over each possible transition
                # prob: probabilty of this outcome
                # next_s: the state we land in
                # reward: the immediate reward received (+1 for Goal, 0 otherwise)
                for prob, next_s, reward, _ in transitions:

                    # populate the probability tensor P
                    # we use += because multiple outcomes might lead to the same next state
                    self.P[s, a, next_s] += prob

                    # populate the reward tensor R
                    self.R[s, a, next_s] = reward


    def _get_policy_coefficients(self, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the policy transition matrix P_pi
        and the expected reward vector r_pi
        for a gven policy using Einstein summation.

        Args:
            policy (np.ndarray): Shape (S, A). The policy probabilities.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - P_pi: Shape (S, S). Transition probabilities between states under this policy.
                - r_pi: Shape (S,). Expected immediate rewards for each state under this policy.
        """
        
        # 1. Compute P_pi (S, S)
        # broadcasting is necessary because policy has shape (S, A) while P has shape (S, A, S)
        P_pi = np.sum(policy[:, :, np.newaxis] * self.P, axis=1)

        # 2. Compute r_pi (S,)
        # First, calculate the expected reward for each specific transition (s, a, s'), 
        # weighing the reward tensor by the probability of that transition happening
        # Shape: (S, A , S)
        expected_rewards_sas = self.P * self.R

        # Next, sum oveer next_states (s') to get expected reward for each pair (s, a)
        # Shape: (S, A)
        expected_rewards_sa = np.sum(expected_rewards_sas, axis=2)

        # Finally, average over actions 'a' weighted by the policy
        # Shape: (S,)
        r_pi = np.sum(expected_rewards_sa * policy, axis=1)

        return P_pi, r_pi

    def _get_policy_coefficients_einsum(self, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the policy transition matrix P_pi
        and the expected reward vector r_pi
        for a gven policy using Einstein summation.

        Args:
            policy (np.ndarray): Shape (S, A). The policy probabilities.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - P_pi: Shape (S, S). Transition probabilities between states under this policy.
                - r_pi: Shape (S,). Expected immediate rewards for each state under this policy.
        """
        
        # 1. Compute P_pi (S, S) by contracting the action dimension 'a' 
        # between the policy and the P tensor
        # 'sa' (policy) and 'sak' (P tensor) -> 'sk' (next state probs given current state)
        # the repetition of 'sa' in both input operands implies that for all (s, a, k),
        # the product policy[s, a] * P[s, a, k] is computed;
        # the omission of 'a' from the output operand implies that the result is summed over all 'a'
        P_pi = np.einsum('sa, sak -> sk', policy, self.P)

        # 2. Compute r_pi (S,) in a single line
        r_pi = np.einsum('sa, sak, sak -> s', policy, self.P, self.R)

        # For clarity, this could be split into two operations:
        # expected_rewards_sa =np.einsum('sak, sak -> sa', self.P, self.R)
        # r_pi = np.einsum('sa, sa -> s', policy, expected_rewards_sa)
        
        return P_pi, r_pi

    
    def policy_evaluation(self, policy: np.ndarray, gamma=0.99, theta=1e-9) -> np.ndarray:
        """
        Performs iterative policy evaluation to find V given the policy.

        Args:
            policy (np.andarray): Shape (S, A). The policy to evaluate.
            gamma (float): Discount factor.
            theta (float): Convergence threshold.

        Returns:
            np.ndarray: Shape (S, ). The value function for the given policy.
        """

        P_pi, r_pi = self._get_policy_coefficients_einsum(policy)
        
        V = np.zeros(self.S)
            
        while True:
            # The Bellman Expectation Equation in vector form:
            # V_new = r_pi + gamma * (P_pi @ V_old)
            # @ symbol denotes matrix multiplication
            V_new = r_pi + gamma * P_pi @ V

            # check for convergence (max absolute difference)
            if np.max(np.abs(V_new - V)) < theta:
                break

            V = V_new

        return V


    def get_q_values(self, V: np.ndarray, gamma=0.99) -> np.ndarray:
        """
        Calculates the Q-value for all state-action pairs given a state-value function V.

        Q(s, a) = Expected Immediate Reward + gamma * Expected Future Value

        Args:
            V (np.ndarray): Shape (S,). The current state-value estimates.
            gamma (float): Discount factor.

        Returns:
            np.ndarray: Shape (S, A). The Q-values for all state-action pairs.
        """

        # Expected Immediate Reward = sum_k P(k|s, a) * R(s, a, k)
        # Expected Future Value = sum_k P(k | s, a) * V(k)
        return np.einsum('sak, sak -> sa', self.P, self.R) + gamma * np.einsum('sak, k -> sa', self.P, V)


    def policy_improvement(self, V: np.ndarray, gamma=0.99) -> np.ndarray:
        """
        Generates a new, greedy policy based on the provided state-value function V.

        Args:
            V (np.ndarray): Shape (S,). Current state-value estimates.
            gamma (float): Discount factor.

        Returns:
            np.ndarray: Shape (S, A). The new greedy, deterministic policy.
        """
        # 1. Calculate Q-values for the current V
        # shape: (S, A)
        Q = self.get_q_values(V, gamma)

        # 2. FInd the best action for each state
        # Shape: (S,) containing indices 0-3 from axis 1 of Q
        best_actions = np.argmax(Q, axis=1)

        # 3. Convert to one-hot encoding
        # We need a policy matrix of shape (S, A) that assigns prob. 1.0
        # to the best action, and 0.0 to all other actions
        new_policy = np.zeros((self.S, self.A))
        # use fancy indexing to set the 1s
        # new_policy[row_indices, col_indices]
        new_policy[np.arange(self.S), best_actions] = 1.0

        return new_policy


    def policy_iteration(self, gamma=0.9, theta=1e-9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the Policy Iteration algorithm.

        1. Start with a random policy.
        2. Evaluate it to get V.
        3. Improve it to get a new greedy policy.
        4. Repeat until the policy becomes stable (does not change anymore).

        Args:
            gamma (float): Discount factor.
            theta (float): Convergence threshold for policy evaluation.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - optimal policy: Shape (S, A)
                - optimal value function: Shape (S,)
        """
        # 1. Initialize an arbitrary policy
        policy = np.ones((self.S, self.A)) / self.A

        iteration_count = 0

        while True:

            iteration_count += 1

            # 2. Policy Evaluation ("prediction step")
            V = self.policy_evaluation(policy, gamma, theta)

            # 3. Policy Improvement ("control step")
            # Create a new policy that is greedy w.r.t. the one just calculated
            new_policy = self.policy_improvement(V, gamma)

            # 4. Check for stability
            # If the new policy is the same as the old one, we are done
            if np.array_equal(new_policy, policy):
                print(f"Policy iteration converged after {iteration_count} step.")
                break

            policy = new_policy

        return policy, V


    def value_iteration(self, gamma=0.99, theta=1e-9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the Value Iteration algorithm to find the optimal policy and value function.

        Unlike policy iteration, which alternates between full evaluation and improvement, 
        value iteration combines them into a single update step:
        V(s) <- max_a Q(s, a)

        Args:
            gamma (float): Discount factor.
            theta (float): Convergence threshold for V.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - optimal policy: Shape (S, A)
                - optimal value function: Shape (S,)
        """

        # 1. Initialize V arbitrarily
        V = np.zeros(self.S)

        iteration_count = 0

        while True:

            iteration_count += 1

            # 2. Compute Q-values and take the max over actions
            V_new = self.get_q_values(V, gamma).max(axis=1)

            # 3. Check for convergence
            if np.max(np.abs(V_new - V)) < theta:
                print(f"Value iteration converged after {iteration_count} step.")
                break

            V = V_new

        # 4. Create a new policy that is greedy w.r.t. the optimal value function just calculated
        optimal_policy = self.policy_improvement(V, gamma)

        return optimal_policy, V