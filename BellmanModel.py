import numpy as np
import gymnasium as gym

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