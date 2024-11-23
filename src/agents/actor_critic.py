import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCriticNet(nn.Module):
    def __init__(self, board_size):
        super(ActorCriticNet, self).__init__()
        action_size = board_size * board_size + 1  # Including pass action

        self.shared_layers = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.actor_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, action_size),
        )

        self.critic_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        shared = self.shared_layers(x)
        policy_logits = self.actor_head(shared)
        value = self.critic_head(shared)
        return policy_logits, value

    def compute_loss(self, states, actions, rewards, policy_targets):
        """
        Compute combined loss for actor-critic training.
        Args:
            states (torch.Tensor): Input board states.
            actions (torch.Tensor): Chosen actions.
            rewards (torch.Tensor): Rewards (game outcomes).
            policy_targets (torch.Tensor): Target policy probabilities from MCTS.
        """
        policy_logits, values = self.forward(states)

        # Policy loss (cross-entropy between predicted and target policies)
        policy_loss = -torch.sum(policy_targets * torch.log_softmax(policy_logits, dim=1), dim=1).mean()

        # Value loss (MSE between predicted and target rewards)
        value_loss = nn.functional.mse_loss(values.squeeze(-1), rewards)

        # Combined loss
        loss = policy_loss + value_loss
        return loss, policy_loss.item(), value_loss.item()

    def train_step(self, optimizer, states, actions, rewards, policy_targets):
        """
        Perform a single training step.
        """
        optimizer.zero_grad()
        loss, policy_loss, value_loss = self.compute_loss(states, actions, rewards, policy_targets)
        loss.backward()
        optimizer.step()
        return loss.item(), policy_loss, value_loss


class ActorCriticPolicy:
    def __init__(self, model, mcts_simulations, temperature=1.0):
        """
        Initialize the Actor-Critic policy.
        Args:
            model (ActorCriticNet): Neural network for policy and value estimation.
            mcts_simulations (int): Number of MCTS simulations.
            temperature (float): Exploration temperature for policy adjustment.
        """
        self.model = model
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature

    def __call__(self, go_env, debug=False):
        """
        Use the policy to select actions.
        Args:
            go_env: GymGo environment.
            debug (bool): Whether to return debug information.
        """
        if self.mcts_simulations > 0:
            # MCTS-guided policy
            from src.agents.mcts import mcts_search
            root_node = mcts_search(go_env, self.model, num_simulations=self.mcts_simulations)
            visit_counts = root_node.get_visit_counts()
            policy = visit_counts ** (1 / self.temperature)
            policy /= policy.sum()

        else:
            # Direct policy from the actor network
            state = go_env.canonical_state()
            policy_logits, _ = self.model(torch.tensor(state[np.newaxis], dtype=torch.float32))
            policy = torch.softmax(policy_logits.squeeze(), dim=0).detach().numpy()

        if debug:
            return policy, root_node if self.mcts_simulations > 0 else None
        return policy

