import os
import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from agents.actor_critic import ActorCriticNetwork
from agents.mcts import mcts_search

class GoTrainer:
    def __init__(self, board_size, mcts_simulations, temperature, lr, batch_size, epochs, model_path):
        self.board_size = board_size
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_path = model_path
        
        self.model = ActorCriticNetwork(board_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def self_play(self, num_games, go_env):
        """
        Generate self-play data using MCTS-guided Actor-Critic policy.

        Args:
            num_games (int): Number of self-play games to generate.
            go_env: GymGo environment.
        Return:
            list of tuples: (state, policy, value) for training.
        """

        training_data = []
        for game in range(num_games):
            go_env.reset()
            done = False
            states, policies, values = [], [], []

            while not done:
                # Perform MCTS to get the polcy
                root_node = mcts_search(go_env, self.model, self.mcts_simulations)
                visit_counts = root_node.get_visit_counts()
                policy = visit_counts ** (1 / self.temperature)
                policy /= policy.sum()

                # Store the state and policy
                canonical_state = go_env.canonical_state()
                states.append(canonical_state)
                policies.append(policy)

                # Select a move based on the policy
                move = np.random.choice(len(policy), p=policy)
                _, _, done, _ = go_env.step(move)

            # Assign values to states based on game outcome
            winner = go_env.winner()
            values = [1 if winner == 1 else -1] * len(states)

            training_data.extend(zip(states, policies, values))

        return training_data
    
    def train(self, training_data):
        """
        Train the Actor-Critic network on the provided training data.

        Args:
            training_data (list of tuples): (state, policy, value) for training.
        """
        
        # Convert data to pytorch tensors
        states = torch.tensor([x[0] for x in training_data], dtype=torch.float32)
        policies = torch.tensor([x[1] for x in training_data], dtype=torch.float32)
        values = torch.tensor([x[2] for x in training_data], dtype=torch.float32)

        dataset = TensorDataset(states, policies, values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            total_loss, total_policy_loss, total_value_loss = 0, 0, 0

            for batch_states, batch_policies, batch_values in dataloader:
                batch_states = batch_states.to(self.device)
                batch_policies = batch_policies.to(self.device)
                batch_values = batch_values.to(self.device)

                # Perform training step
                loss, policy_loss, value_loss = self.model.train_step(self.optimizer, batch_states, None, batch_values, batch_policies)

                total_loss += loss
                total_policy_loss += policy_loss
                total_value_loss += value_loss

    def evaluate(self, go_env, num_games=10):
        """
        Evaluate current model.
        Args:
            go_env: GymGo environment
            num_games: Number of games played to evaluate performance
        Returns:
            float: winrate of model
        
        """


