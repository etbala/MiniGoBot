import unittest
import numpy as np
import torch
from unittest.mock import MagicMock

from go_bot.mcts import Node, expand_and_evaluate, mcts_step, mcts_search


class TestMCTS(unittest.TestCase):
    def setUp(self):
        """
        Set up mock objects for testing.
        """
        # Mock GymGo environment
        self.mock_env = MagicMock()
        self.mock_env.valid_moves.return_value = [1, 0, 1, 1, 0]
        self.mock_env.children.return_value = [
            np.array([[0, 0], [0, 1]]),  # Child 1
            np.array([[1, 0], [0, 0]]),  # Child 2
            np.array([[0, 1], [0, 0]]),  # Child 3
        ]
        self.mock_env.game_ended.return_value = False
        self.mock_env.winning.return_value = 0

        # Mock Actor-Critic network
        self.mock_actor_critic = MagicMock()
        self.mock_actor_critic.return_value = (
            torch.tensor([[0.2, 0.5, 0.1, 0.2, 0.0]]),  # Policy logits as tensor
            torch.tensor([[0.3]])  # Value as tensor
        )

        # Mock the parameters method to return a new iterator each time
        param = torch.nn.Parameter(torch.empty(0))
        self.mock_actor_critic.parameters.side_effect = lambda: iter([param])

    def test_node_initialization(self):
        """
        Test Node initialization and basic properties.
        """
        state = np.array([[0, 1], [1, 0]])
        node = Node(state)

        self.assertEqual(node.state.tolist(), state.tolist())
        self.assertIsNone(node.parent)
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.value_sum, 0)
        self.assertTrue(node.is_leaf())
        self.assertIsNone(node.prior)

    def test_node_expand(self):
        """
        Test Node expansion with valid moves and child states.
        """
        state = np.array([[0, 1], [1, 0]])
        node = Node(state)

        # Expand node using mocked environment and policy
        policy = np.array([0.1, 0.2, 0.4, 0.3, 0.0])
        node.expand(self.mock_env, policy)

        # Check that child nodes were created for valid moves
        self.assertEqual(len(node.child_nodes), 3)
        for move in [0, 2, 3]:
            self.assertIn(move, node.child_nodes)
            self.assertIsInstance(node.child_nodes[move], Node)

        # Check that priors match the policy
        for move, child in node.child_nodes.items():
            self.assertEqual(child.prior, policy[move])

    def test_expand_and_evaluate(self):
        """
        Test expand_and_evaluate functionality.
        """
        state = np.array([[0, 1], [1, 0]])
        node = Node(state)

        value = expand_and_evaluate(node, self.mock_actor_critic, self.mock_env)

        # Verify that the value is returned correctly
        self.assertAlmostEqual(value, 0.3)

        # Verify that expand was called with the correct arguments
        self.assertEqual(len(node.child_nodes), 3)
        for move in [0, 2, 3]:
            self.assertIn(move, node.child_nodes)

    def test_mcts_step(self):
        """
        Test a single MCTS step.
        """
        state = np.array([[0, 1], [1, 0]])
        root = Node(state)

        # Run one MCTS step
        mcts_step(root, self.mock_actor_critic, self.mock_env)

        # Check that root node visits and child nodes were updated
        self.assertGreater(root.visits, 0)
        self.assertGreater(len(root.child_nodes), 0)

    def test_mcts_search(self):
        """
        Test full MCTS search functionality.
        """

        # Run MCTS search
        root = mcts_search(self.mock_env, self.mock_actor_critic, num_simulations=10)

        # Verify root node properties after search
        self.assertEqual(root.visits, 10)
        self.assertGreater(len(root.child_nodes), 0)

        # Verify that child nodes have visit counts
        for child in root.child_nodes.values():
            self.assertGreaterEqual(child.visits, 0)

    def test_node_expand_with_no_valid_moves(self):
        """Test Node.expand when there are no valid moves."""
        state = np.array([[0, 1], [1, 0]])
        node = Node(state)

        # Mock valid_moves to be all zeros (no valid moves)
        self.mock_env.valid_moves.return_value = [0, 0, 0, 0, 0]
        policy = np.array([0.1, 0.2, 0.3, 0.4, 0.0])

        node.expand(self.mock_env, policy)
        self.assertEqual(len(node.child_nodes), 0)  # No children should be created

    def test_node_expand_with_terminal_states(self):
        """Test Node.expand when all child states are terminal."""
        state = np.array([[0, 1], [1, 0]])
        node = Node(state)

        # Mock game_ended to return True for all child states
        self.mock_env.game_ended.side_effect = lambda s: True
        policy = np.array([0.1, 0.2, 0.3, 0.4, 0.0])

        node.expand(self.mock_env, policy)
        for move, child in node.child_nodes.items():
            self.assertTrue(self.mock_env.game_ended(child.state))

    def test_mcts_with_one_simulation(self):
        """Test MCTS with only one simulation."""
        state = np.array([[0, 1], [1, 0]])
        self.mock_env.canonical_state.return_value = state

        root = mcts_search(self.mock_env, self.mock_actor_critic, num_simulations=1)
        self.assertEqual(root.visits, 1)  # Root should have exactly one visit
        for child in root.child_nodes.values():
            self.assertGreaterEqual(child.visits, 0)

    def test_mcts_with_invalid_moves_only(self):
        """Test MCTS with all moves invalid."""
        state = np.array([[0, 1], [1, 0]])
        self.mock_env.canonical_state.return_value = state

        # Mock valid_moves to be all zeros
        self.mock_env.valid_moves.return_value = [0, 0, 0, 0, 0]

        root = mcts_search(self.mock_env, self.mock_actor_critic, num_simulations=10)
        self.assertEqual(len(root.child_nodes), 0)  # No children should be created


if __name__ == "__main__":
    unittest.main()
