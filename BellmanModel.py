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