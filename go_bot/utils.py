import argparse
import datetime
import logging
import os

import numpy as np
import torch

from go_bot import self_play
from go_bot.actor_critic import get_modelpath
from go_bot import baselines


def hyperparameters(args_encoding=None):
    today = str(datetime.date.today())

    parser = argparse.ArgumentParser()
    # Go Environment
    parser.add_argument('--size', type=int, default=9, help='board size')
    parser.add_argument('--reward', type=str, choices=['real', 'heuristic'], default='heuristic', help='reward system')

    # Params
    parser.add_argument('--mcts', type=int, default=0, help='monte carlo searches (actor critic)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--temp', type=float, default=1, help='initial temperature')

    # Data Sizes
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--replaysize', type=int, default=16, help='max number of games to store')
    parser.add_argument('--batches', type=int, default=1000, help='number of batches to train on for one iteration')

    # Loading
    parser.add_argument('--customdir', type=str, default='', help='load model from custom directory')
    parser.add_argument('--latest-checkpoint', type=bool, default=False, help='load model from checkpoint')

    # Training
    parser.add_argument('--iterations', type=int, default=128, help='iterations')
    parser.add_argument('--episodes', type=int, default=32, help='episodes')
    parser.add_argument('--evaluations', type=int, default=16, help='episodes')
    parser.add_argument('--eval-interval', type=int, default=2, help='iterations per evaluation')

    # Disk Data
    parser.add_argument('--replay-path', type=str, default='bin/replay.pickle', help='path to store replay')
    parser.add_argument('--checkdir', type=str, default=f'bin/checkpoints/{today}/')

    # Model
    parser.add_argument('--model', type=str, choices=['ac', 'rand', 'greedy', 'human'],
                        default='ac', help='type of model')

    # Hardware
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='device for pytorch models')

    # Other
    parser.add_argument('--render', type=str, choices=['terminal', 'human'], default='terminal',
                        help='type of rendering')

    args = parser.parse_args(args_encoding)

    if args.device == "cuda":
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setattr(args, 'checkpath', os.path.join(args.checkdir, 'checkpoint.pt'))
    return args


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def config_log(args):
    formatter = logging.Formatter('%(message)s')

    # Use append mode if latest-checkpoint is True
    file_mode = 'a' if args.latest_checkpoint else 'w'
    handler = logging.FileHandler(os.path.join(args.checkdir, 'training_log.txt'), file_mode)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[console, handler])


def log_info(s):
    logging.info(s)


def log_debug(s):
    logging.debug(s)


def sync_checkpoint(args, new_pi, old_pi):
    checkpath = get_modelpath(args)
    torch.save(new_pi.pt_model.state_dict(), checkpath)
    old_pi.pt_model.load_state_dict(torch.load(checkpath, map_location=args.device))


def play_games(go_env, pi1, pi2, requested_episodes):
    p1wr, black_wr, replay, steps = self_play.play_games(go_env, pi1, pi2, requested_episodes, progress=True)
    log_debug(f'{pi1} V {pi2} | {requested_episodes} GAMES, {100 * p1wr:.1f}% WIN({100 * black_wr:.1f}% BLACK_WIN)')
    return p1wr, black_wr, replay


def get_iter_header():
    return "TIME\tITR\tREPLAY\tC_ACC\tC_LOSS\tA_ACC\tA_LOSS\tG_LOSS\tC_WR\tR_WR\tG_WR"


def get_iter_entry(starttime, iteration, replay_len, metrics, winrates, checkpoint_pi):
    currtime = datetime.datetime.now()
    delta = currtime - starttime
    iter_info = f"{str(delta).split('.')[0]}\t{iteration:02d}\t{replay_len:07d}\t"
    if metrics.crit_acc:
        iter_info += f"{100 * metrics.crit_acc:04.1f}\t{metrics.crit_loss:04.3f}\t"
    else:
        iter_info += "____\t____\t"
    if metrics.act_acc:
        iter_info += f"{100 * metrics.act_acc:04.1f}\t{metrics.act_loss:04.3f}\t"
    else:
        iter_info += "____\t____\t"
    iter_info += f"{metrics.game_loss:04.3f}\t" if metrics.game_loss else "____\t"
    iter_info += f"{100 * winrates[checkpoint_pi]:04.1f}\t{100 * winrates[baselines.RAND_PI]:04.1f}"
    return iter_info
