import os
import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from go_bot.actor_critic import ActorCriticNet
from go_bot.mcts import mcts_search

class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.children = []
        self.pis = []

    def get_events(self):
        events = []
        black_won = self.get_winner()
        n = len(self)
        for i, (state, action, reward, children, pi) in enumerate(
            zip(self.states, self.actions, self.rewards, self.children, self.pis)
        ):
            turn = i % 2
            won = black_won if turn == 0 else -black_won
            terminal = i == n - 1
            events.append((state, action, reward, children, terminal, won, pi))
        return events

    def add_event(self, state, action, reward, children, pi):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.children.append(children)
        self.pis.append(pi)

    def set_win(self, black_won):
        self.rewards[-1] = black_won

    def get_winner(self):
        return self.rewards[-1]

    def __len__(self):
        assert all(len(lst) == len(self.states) for lst in [self.actions, self.rewards, self.children, self.pis])
        return len(self.states)

def pit(go_env, black_policy, white_policy):
    traj = Trajectory()
    num_steps = 0
    state = go_env.canonical_state()
    max_steps = 2 * (go_env.size ** 2)
    done = False

    while not done:
        # Determine the current player's policy
        curr_turn = go_env.turn()
        pi = black_policy(go_env) if curr_turn == 1 else white_policy(go_env)

        # Select an action based on the policy
        action = np.random.choice(len(pi), p=pi)

        # Take a step in the environment
        padded_children = go_env.children(canonical=True, padded=True)
        _, reward, done, _ = go_env.step(action)

        # Enforce maximum steps
        if num_steps >= max_steps:
            done = True

        # Log the event
        traj.add_event(state, action, reward, padded_children, pi)

        # Prepare for the next turn
        state = padded_children[action]
        num_steps += 1

    # Determine the winner and update trajectory
    black_won = go_env.winning()
    traj.set_win(black_won)

    return black_won, num_steps, traj

def play_games(go_env, first_policy, second_policy, episodes, progress=True):
    replay = []
    all_steps = []
    first_wins, black_wins = 0, 0

    pbar = tqdm(range(episodes), desc="Playing games", disable=not progress)
    for i in pbar:
        go_env.reset()
        if i % 2 == 0:
            black_won, steps, traj = pit(go_env, first_policy, second_policy)
            first_won = black_won
        else:
            black_won, steps, traj = pit(go_env, second_policy, first_policy)
            first_won = -black_won

        black_wins += black_won == 1
        first_wins += first_won == 1
        all_steps.append(steps)
        replay.append(traj)

        if progress:
            pbar.set_postfix(win_rate=f"{100 * first_wins / (i + 1):.1f}%")

    return first_wins / episodes, black_wins / episodes, replay, all_steps
