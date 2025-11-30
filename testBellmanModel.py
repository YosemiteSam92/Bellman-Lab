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

    """ Test policy iteration """

    def test_policy_iteration_convergence(self):
        """
        Verifies that policy iteration produces a valid policy and value function.
        """
        optimal_policy, optimal_V = self.model.policy_iteration()

        # --- check shapes ---
        self.assertEqual(
            optimal_policy.shape,
            (16, 4),
            f"Policy tensor shape mismatch. Should be (16, 4), got {optimal_policy.shape}"
        )

        self.assertEqual(
            optimal_V.shape,
            (16, ),
            f"V tensor shape mismatch. Should be (16,), got {optimal_V.shape}"
        )

        # --- check that the policy is deterministic (one-hot) ---
        # summing over actions should yield 1.0 for every state
        self.assertTrue(
            np.allclose(np.sum(optimal_policy, axis=1), 1.0),
            "Policy is not normalized to 1 over actions."
        )
        # all the probability mass should be concentrated onto the optimal action,
        # with all other actions being assigned zero probability
        self.assertTrue(
            np.allclose(np.max(optimal_policy, axis=1), 1.0),
            "Policy does not assign all probability mass to the optimal action."
        )

        # --- Sanity check: the start state (0) should have a value > 0 ---
        # because it is possible to reach the goal (15) which has reward +1
        self.assertGreater(
            optimal_V[0],
            0.0,
            "The start state (0) should have a value > 0."
        )

        # --- Sanity check: the goal state (15) should have zero value ---
        # value of a state = expected immediate reward + gamma * expected future value
        # since the goal state is terminal, its expected guture reward is 0
        # furthermore, in this gymnasium implementation, terminal states transition to themselves
        # with zero reward, so the expected immediate reward is also 0
        self.assertEqual(
            optimal_V[15],
            0.0,
            "The goal state (15) should have zero value."
        )

    """ Test value iteration """

    def test_value_iteration_optimality(self):
        """
        Verifies that Value Iteration converges to the same result as Policy Iteration.
        """
        gamma = 0.99

        # 1. Run policy iteration (ground truth)
        pi_policy, pi_V = self.model.policy_iteration(gamma=gamma)

        # 2. Run value iteration
        vi_policy, vi_V = self.model.value_iteration(gamma=gamma)

        # 3. Compare value functions
        # They should be identical within floating-point tolerance
        self.assertTrue(
            np.allclose(pi_V, vi_V, atol=1e-5),
            f"Value Iteration V* does not match Policy Iteration V*.\nDiff:\n {pi_V - vi_V}"
        )

        # 4. Compare policies
        self.assertTrue(
            np.allclose(pi_policy, vi_policy, atol=1e-5),
            f"Policy Iteration policy does not match Value Iteration policy.\nDiff: {pi_policy - vi_policy}"
        )


if __name__ == "__main__":
    unittest.main()