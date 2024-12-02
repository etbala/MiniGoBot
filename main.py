import collections
import os
from datetime import datetime

import gym
import torch

from go_bot import data, utils
from go_bot import baselines
from go_bot import actor_critic


def model_eval(args, curr_pi, checkpoint_pi, winrates):
    go_env = gym.make('gym_go:go-v0', size=args.size, reward_method=args.reward)
    for opponent in [checkpoint_pi, baselines.RAND_PI]:
        utils.log_debug(f'Pitting {curr_pi} V {opponent}')
        wr, _, _ = utils.play_games(go_env, curr_pi, opponent, args.evaluations)
        winrates[opponent] = wr


def train_step(args, curr_pi, optim, checkpoint_pi):
    go_env = gym.make('gym_go:go-v0', size=args.size, reward_method=args.reward)
    curr_model = curr_pi.pt_model

    utils.log_debug(f'Self-Playing {checkpoint_pi} V {checkpoint_pi}...')
    _, _, replays = utils.play_games(go_env, checkpoint_pi, checkpoint_pi, args.episodes)

    data.append_replay(args, replays)
    utils.log_debug('Added all replay data to disk')

    traindata, replay_len = data.sample_eventdata(args.replay_path, args.batches, args.batchsize)

    utils.log_debug(f'Optimizing in {len(traindata)} training steps...')
    metrics = curr_model.optimize(traindata, optim)

    utils.log_debug(f'Optimized | {str(metrics)}')
    return metrics, replay_len


def train(args, curr_pi, checkpoint_pi):
    curr_model = curr_pi.pt_model
    optim = torch.optim.Adam(curr_model.parameters(), args.lr, weight_decay=1e-4)
    starttime = datetime.now()

    utils.log_info(utils.get_iter_header())
    winrates = collections.defaultdict(float)

    for iteration in range(args.iterations):
        metrics, replay_len = train_step(args, curr_pi, optim, checkpoint_pi)

        if (iteration + 1) % args.eval_interval == 0:
            model_eval(args, curr_pi, checkpoint_pi, winrates)

        utils.sync_checkpoint(args, curr_pi, checkpoint_pi)

        iter_info = utils.get_iter_entry(starttime, iteration, replay_len, metrics, winrates, checkpoint_pi)
        utils.log_info(iter_info)


if __name__ == '__main__':
    args = utils.hyperparameters()

    if not os.path.exists(args.checkdir):
        os.makedirs(args.checkdir, exist_ok=True)

    utils.config_log(args)
    utils.log_debug(f"Args: {args}")

    curr_model = actor_critic.ActorCriticNet(args.size)
    curr_pi = actor_critic.ActorCriticPolicy('Current', curr_model, args)
    checkpoint_model = actor_critic.ActorCriticNet(args.size)
    checkpoint_pi = actor_critic.ActorCriticPolicy('Checkpoint', checkpoint_model, args)

    utils.log_debug(f'Model has {utils.count_parameters(curr_model):,} trainable parameters')

    device = torch.device(args.device)
    curr_model.to(device)
    checkpoint_model.to(device)

    train(args, curr_pi, checkpoint_pi)
