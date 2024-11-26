import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from go_bot.mcts import mcts_search

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

    def compute_loss(self, states, actions, rewards, policy_targets, valid_moves):
        """
        Compute combined loss for actor-critic training.
        Args:
            states (torch.Tensor): Input board states.
            actions (torch.Tensor): Chosen actions.
            rewards (torch.Tensor): Rewards (game outcomes).
            policy_targets (torch.Tensor): Target policy probabilities from MCTS.
        """
        device = next(self.parameters()).device
        states = states.to(device)
        rewards = rewards.to(device)
        policy_targets = policy_targets.to(device)
        valid_moves = valid_moves.to(device)

        policy_logits, values = self.forward(states)

        # Debugging print statements
        print(f"policy_logits shape: {policy_logits.shape}")  # Should be [batch_size, action_size]
        print(f"valid_moves shape: {valid_moves.shape}")      # Should be [batch_size, action_size]

        # Mask invalid moves in logits
        large_negative = -1e8
        policy_logits = policy_logits * valid_moves + (1 - valid_moves) * large_negative


        # Adjust policy targets
        policy_targets = policy_targets * valid_moves
        policy_targets /= policy_targets.sum(dim=1, keepdim=True) + 1e-8

        # Compute log probabilities
        log_probs = torch.log_softmax(policy_logits, dim=1)

        # Policy loss (cross-entropy)
        policy_loss = -torch.sum(policy_targets * log_probs, dim=1).mean()

        # Value loss (MSE)
        value_loss = nn.functional.mse_loss(values.squeeze(-1), rewards)

        # Combined loss
        loss = policy_loss + value_loss
        return loss, policy_loss.item(), value_loss.item()

    def train_step(self, optimizer, states, actions, rewards, policy_targets, valid_moves):
        """
        Perform a single training step.
        """
        optimizer.zero_grad()
        loss, policy_loss, value_loss = self.compute_loss(states, actions, rewards, policy_targets, valid_moves)
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
            root_node = mcts_search(go_env, self.model, num_simulations=self.mcts_simulations)
            total_visits = sum(child.visits for child in root_node.child_nodes.values())
            policy = np.zeros(go_env.action_space.n)
            for move, child in root_node.child_nodes.items():
                policy[move] = child.visits / total_visits if total_visits > 0 else 0
            policy = policy ** (1 / self.temperature)
            policy /= policy.sum()
        else:
            state = go_env.canonical_state()
            device = next(self.model.parameters()).device
            state_tensor = torch.tensor(state[np.newaxis], dtype=torch.float32).to(device)
            policy_logits, _ = self.model(state_tensor)
            policy_logits = policy_logits.squeeze(0)

            # Get valid moves
            valid_moves = go_env.valid_moves()
            valid_moves = torch.tensor(valid_moves, dtype=torch.bool).to(device)

            # Mask invalid moves
            large_negative = -1e8
            policy_logits[~valid_moves] = large_negative

            # Apply softmax
            policy = torch.softmax(policy_logits, dim=0).detach().cpu().numpy()

        if debug:
            return policy, root_node if self.mcts_simulations > 0 else None
        return policy
