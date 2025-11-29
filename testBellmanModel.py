import unittest
import numpy as np
import gymnasium as gym
from BellmanModel import BellmanModel

class TestBellmanModel(unittest.TestCase):

    def setUp(self):
        """
        Runs before every test, setting up a fresh environment and model.
        """
        self.env = gym.make("FrozenLake-v1", is_slippery=True)
        self.model = BellmanModel(self.env)

    """ Test tensor-ification methods """

    def test_tensor_shapes(self):
        """
        Verifies that P and R tensors have the correct (S,A,S) shapes.
        """
        expected_shape = (16, 4, 16) # for 4x4 FrozenLake
        
        self.assertEqual(
            self.model.P.shape,
            expected_shape,
            f"P tensor shape mismatch. Got {self.model.P.shape}"
        )

        self.assertEqual(
            self.model.R.shape,
            expected_shape,
            f"R tensor shape mismatch. Got {self.model.R.shape}"
        )

    def test_probability_distribution(self):
        """
        Verifies that for every state-action pair, the probabilities of 
        moving to the next states sum to 1.0 (normalization).
        """

        # sum P over the last axis (next_states)
        # resulting shape (S, A) = (16, 4)
        prob_sums = np.sum(self.model.P, axis=2)

        # check that all sums are close to 1.0
        # using np.allclose because of floating point errors
        self.assertTrue(
            np.allclose(prob_sums, 1.0),
            "Probabilities do not sum to 1 for all state-action pairs."
        )

    def test_goal_reward(self):
        """
        Verifies that the reward tensor contains the goal reward (+1)
        """

        # The reward should be present in all transitions leading to the 
        # goal state (15). We verify that the max reward in the tensor is 1.0
        self.assertEqual(
            self.model.R.max(),
            1.0,
            "Reward tensor does not contain the goal reward (+1)."
        )

    """ Test policy evaluation """

    def test_policy_evaluation_runs(self):
        """
        Verifies that policy_evaluation runs and returns a 
        state-value vector of correct shape (S,).
        """
        # Create a uniform random policy: 1/4 prob for all 4 actions in all 16 states
        policy = np.ones((16, 4)) / 4.0

        # Run evaluation
        V_pi = self.model.policy_evaluation(policy)

        # Check that V_pi has the correct shape
        expected_shape = (16,) # for 4x4 FrozenLake
        self.assertEqual(
            V_pi.shape,
            expected_shape,
            f"V_pi tensor shape mismatch. Should be (16,), got {V_pi.shape}"
        )

        # Check state values are non-negative for all states
        self.assertTrue(np.all(V_pi >= 0), "All values in V_pi should be non-negative.")

if __name__ == "__main__":
    unittest.main()