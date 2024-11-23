import unittest
import torch
from torch.optim import Adam
from src.agents.actor_critic import ActorCriticNet


class TestActorCriticNet(unittest.TestCase):
    def setUp(self):
        """
        Set up the Actor-Critic network and mock data for testing.
        """
        self.board_size = 19
        self.action_size = self.board_size * self.board_size + 1
        self.model = ActorCriticNet(self.board_size)

        # Mock data
        self.batch_size = 32
        self.states = torch.rand(self.batch_size, 6, self.board_size, self.board_size)  # Random states
        self.actions = torch.randint(0, self.action_size, (self.batch_size,))  # Random actions
        self.rewards = torch.rand(self.batch_size)  # Random rewards
        self.policy_targets = torch.rand(self.batch_size, self.action_size)  # Random policy targets
        self.policy_targets /= self.policy_targets.sum(dim=1, keepdim=True)  # Normalize to probabilities

    def test_forward_pass(self):
        """
        Test the forward pass of the Actor-Critic network.
        """
        policy_logits, value = self.model(self.states)
        self.assertEqual(policy_logits.shape, (self.batch_size, self.action_size))
        self.assertEqual(value.shape, (self.batch_size, 1))
        self.assertFalse(torch.isnan(policy_logits).any())
        self.assertFalse(torch.isnan(value).any())

    def test_compute_loss(self):
        """
        Test the compute_loss method.
        """
        loss, policy_loss, value_loss = self.model.compute_loss(
            self.states, self.actions, self.rewards, self.policy_targets
        )
        self.assertGreater(loss, 0)
        self.assertGreater(policy_loss, 0)
        self.assertGreater(value_loss, 0)

    def test_train_step(self):
        """
        Test the train_step method.
        """
        optimizer = Adam(self.model.parameters(), lr=0.001)

        # Perform a training step
        loss, policy_loss, value_loss = self.model.train_step(
            optimizer, self.states, self.actions, self.rewards, self.policy_targets
        )
        self.assertGreater(loss, 0)
        self.assertGreater(policy_loss, 0)
        self.assertGreater(value_loss, 0)

    def test_batch_size_invariance(self):
        """
        Test the network with different batch sizes.
        """
        for batch_size in [1, 8, 64]:
            states = torch.rand(batch_size, 6, self.board_size, self.board_size)
            policy_logits, value = self.model(states)
            self.assertEqual(policy_logits.shape, (batch_size, self.action_size))
            self.assertEqual(value.shape, (batch_size, 1))

    def test_value_loss_behavior(self):
        """
        Test the value loss behavior with extreme rewards.
        """
        extreme_rewards = torch.tensor([100.0] * self.batch_size)  # High positive rewards
        loss, policy_loss, value_loss = self.model.compute_loss(
            self.states, self.actions, extreme_rewards, self.policy_targets
        )
        self.assertGreater(value_loss, 0)

        extreme_rewards = torch.tensor([-100.0] * self.batch_size)  # High negative rewards
        loss, policy_loss, value_loss = self.model.compute_loss(
            self.states, self.actions, extreme_rewards, self.policy_targets
        )
        self.assertGreater(value_loss, 0)


if __name__ == "__main__":
    unittest.main()
