import collections
import os
import pickle
import random

import gym
import numpy as np

go_env = gym.make('gym_go:go-v0', size=0)
GoVars = go_env.govars
GoGame = go_env.gogame


def batch_invalid_moves(states):
    """
    Returns 1's where moves are invalid and 0's where moves are valid
    """
    assert len(states.shape) == 4
    batchsize = states.shape[0]
    board_size = states.shape[2]
    invalid_moves = states[:, GoVars.INVD_CHNL].reshape((batchsize, -1))
    invalid_moves = np.insert(invalid_moves, board_size ** 2, 0, axis=1)
    return invalid_moves


def batch_valid_moves(states):
    return 1 - batch_invalid_moves(states)


def batch_invalid_values(states):
    """
    Returns the action values of the states where invalid moves have -infinity value (minimum value of float32)
    and valid moves have 0 value
    """
    invalid_moves = batch_invalid_moves(states)
    invalid_values = np.finfo(np.float32).min * invalid_moves
    return invalid_values


def batch_win_children(batch_children):
    batch_win = []
    for children in batch_children:
        win = []
        for state in children:
            if GoGame.game_ended(state):
                win.append(GoGame.winning(state))
            else:
                win.append(0)
        batch_win.append(win)
    return np.array(batch_win)


def batch_padded_children(states):
    all_children = []
    all_valid_moves = batch_valid_moves(states)
    for state, valid_moves in zip(states, all_valid_moves):
        children = GoGame.children(state, canonical=True, padded=True)
        all_children.append(children)
    return all_children


def batch_random_symmetries(states):
    assert len(states.shape) == 4
    processed_states = []
    for state in states:
        processed_states.append(GoGame.random_symmetry(state))
    return np.array(processed_states)

# Actual Symmetries:
# def batch_random_symmetries(states):
#     # Generate random transformations for each state
#     flips = np.random.choice([True, False], size=states.shape[0])
#     rotations = np.random.choice([0, 1, 2, 3], size=states.shape[0])
#     transformed_states = []
#     for state, flip, rot in zip(states, flips, rotations):
#         transformed_state = np.copy(state)
#         if flip:
#             transformed_state = np.flip(transformed_state, axis=2)  # Flip horizontally
#         transformed_state = np.rot90(transformed_state, k=rot, axes=(1, 2))
#         transformed_states.append(transformed_state)
#     return np.array(transformed_states)

def batch_combine_state_actions(states, actions):
    new_shape = np.array(states.shape)
    new_shape[1] = 7
    size = new_shape[-1]
    state_actions = np.zeros(new_shape)
    state_actions[:, :-1] = states
    for i, a in enumerate(actions):
        if a < size ** 2:
            r, c = a // size, a % size
            state_actions[i, -1, r, c] = 1

    return state_actions


def replay_to_events(replay):
    trans_trajs = []
    for traj in replay:
        trans_trajs.extend(traj.get_events())
    return trans_trajs


def events_to_numpy(events):
    if len(events) == 0:
        return [], [], [], [], [], []
    unzipped = list(zip(*events))

    states = np.array(list(unzipped[0]), dtype=np.float32)
    actions = np.array(list(unzipped[1]), dtype=np.int64)
    rewards = np.array(list(unzipped[2]), dtype=np.float32).reshape((-1,))
    next_states = np.array(list(unzipped[3]), dtype=np.float32)
    terminals = np.array(list(unzipped[4]), dtype=np.uint8)
    wins = np.array(list(unzipped[5]), dtype=np.int64)
    pis = np.array(list(unzipped[6]), dtype=np.float32)

    return states, actions, rewards, next_states, terminals, wins, pis


def load_replay(replay_path):
    """
    Loads replay data from a directory.
    :param replay_path:
    :param worker_rank: If specified, loads only that specific worker's data. Otherwise it loads all data from all workers
    :return:
    """
    with open(replay_path, 'rb') as f:
        replay = pickle.load(f)
    return replay


def sample_eventdata(replay_path, batches, batchsize):
    replay = load_replay(replay_path)
    replay_len = len(replay)

    black_wins = list(filter(lambda traj: traj.get_winner() == 1, replay))
    black_nonwins = list(filter(lambda traj: traj.get_winner() != 1, replay))
    black_wins = replay_to_events(black_wins)
    black_nonwins = replay_to_events(black_nonwins)

    n = min(len(black_wins), len(black_nonwins))
    sample_size = min(batchsize * batches // 2, n)
    sample_data = random.sample(black_wins, sample_size) + random.sample(black_nonwins, sample_size)

    random.shuffle(sample_data)
    sample_data = events_to_numpy(sample_data)

    sample_size = len(sample_data[0])
    for component in sample_data:
        assert len(component) == sample_size

    splits = max(sample_size // batchsize, 1)
    batched_sampledata = [np.array_split(component, splits) for component in sample_data]
    batched_sampledata = list(zip(*batched_sampledata))

    return batched_sampledata, replay_len


def append_replay(args, replays):
    if os.path.exists(args.replay_path):
        all_replays = load_replay(args.replay_path)
        all_replays.extend(replays)
    else:
        all_replays = replays

    with open(args.replay_path, 'wb') as f:
        pickle.dump(all_replays, f)


def reset_replay(args):
    if os.path.exists(args.replay_path):
        os.remove(args.replay_path)
    replay_buffer = collections.deque(maxlen=args.replaysize)
    with open(args.replay_path, 'wb') as f:
        pickle.dump(replay_buffer, f)
