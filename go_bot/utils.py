import argparse
import datetime
import logging
import math
import multiprocessing as mp
import os
import pickle
import time
from datetime import datetime as dt

import gym
import numpy as np
import torch
from mpi4py import MPI

from go_bot import self_play
from go_bot import data
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
    parser.add_argument('--replaysize', type=int, default=64, help='max number of games to store')
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

    setattr(args, 'checkpath', os.path.join(args.checkdir, f'{args.model}{args.size}.pt'))
    setattr(args, 'custompath', os.path.join(args.customdir, f'{args.model}{args.size}.pt'))

    return args


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def config_log(args=None):
    bare_frmtr = logging.Formatter('%(message)s')
    if args is None:
        handler = logging.Handler()
    else:
        handler = logging.FileHandler(os.path.join(args.checkdir, f'{args.model}{args.size}_stats.txt'), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(bare_frmtr)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(bare_frmtr)
    rootlogger = logging.getLogger()
    rootlogger.setLevel(logging.DEBUG)
    rootlogger.addHandler(console)
    rootlogger.addHandler(handler)

    logging.getLogger('matplotlib.font_manager').disabled = True


def log_info(s):
    logging.info(s)


def log_debug(s):
    s = f"{time.strftime('%H:%M:%S', time.localtime())}\t{s}"
    logging.debug(s)


def mpi_config_log(args, comm: MPI.Intracomm):
    if comm.Get_rank() == 0:
        config_log(args)

    comm.Barrier()


def mpi_log_info(comm: MPI.Intracomm, s):
    """
    Only the first worker prints stuff
    :param rank:
    :param s:
    :return:
    """
    rank = comm.Get_rank()
    if rank == 0:
        log_info(s)


def mpi_log_debug(comm: MPI.Intracomm, s):
    """
    Only the first worker prints stuff
    :param rank:
    :param s:
    :return:
    """
    rank = comm.Get_rank()
    if rank == 0:
        log_debug(s)


def mpi_sync_checkpoint(comm: MPI.Intracomm, args, new_pi, old_pi):
    rank = comm.Get_rank()
    checkpath = get_modelpath(args, 'checkpoint')
    if rank == 0:
        torch.save(new_pi.pt_model.state_dict(), checkpath)
    comm.Barrier()
    # Update other policy
    old_pi.pt_model.load_state_dict(torch.load(checkpath, args.device))


def mpi_sync_data(comm: MPI.Intracomm, args):
    rank = comm.Get_rank()
    if rank == 0:
        # Clear worker data
        data.reset_replay(args)

        if args.customdir != '':
            if os.path.exists(args.checkpath):
                # Ensure that we do not use any old parameters
                os.remove(args.checkpath)
            else:
                mpi_log_debug(comm, f"Starting from model file: {args.custompath}")
        else:
            # Save new model
            _, new_model = baselines.create_policy(args, '')
            torch.save(new_model.state_dict(), args.checkpath)
            mpi_log_debug(comm, "Starting from scratch")

    comm.Barrier()


def mpi_play(comm: MPI.Intracomm, go_env, pi1, pi2, requested_episodes):
    world_size = comm.Get_size()

    worker_episodes = int(math.ceil(requested_episodes / world_size))
    episodes = worker_episodes * world_size
    single_worker = comm.Get_size() <= 1

    timestart = time.time()
    p1wr, black_wr, replay, steps = self_play.play_games(go_env, pi1, pi2, worker_episodes, progress=single_worker)
    timeend = time.time()

    duration = timeend - timestart
    avg_time = comm.allreduce(duration / worker_episodes, op=MPI.SUM) / world_size
    p1wr = comm.allreduce(p1wr, op=MPI.SUM) / world_size
    black_wr = comm.allreduce(black_wr, op=MPI.SUM) / world_size
    avg_steps = comm.allreduce(sum(steps), op=MPI.SUM) / episodes

    mpi_log_debug(comm, f'{pi1} V {pi2} | {episodes} GAMES, {avg_time:.1f} SEC/GAME, {avg_steps:.0f} STEPS/GAME, '
                        f'{100 * p1wr:.1f}% WIN({100 * black_wr:.1f}% BLACK_WIN)')
    return p1wr, black_wr, replay


def multi_proc_play(args1, args2, requested_episodes, workers=4):
    world_size = workers

    worker_episodes = int(math.ceil(requested_episodes / world_size))
    episodes = worker_episodes * world_size

    # Clear replay on disk
    assert args1.replay_path == args2.replay_path
    if os.path.exists(args1.replay_path):
        os.remove(args1.replay_path)

    context = mp.get_context('spawn')
    barrier = context.Barrier(workers)
    queue = context.Queue()
    processes = []
    for rank in range(workers):
        p = context.Process(target=worker_play, args=(rank, queue, barrier, args1, args2, worker_episodes))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    p1wrs = []
    black_wrs = []
    total_durations = []
    total_steps = []
    while not queue.empty():
        p1wr, black_wr, steps, duration = queue.get()
        p1wrs.append(p1wr)
        black_wrs.append(black_wr)
        total_durations.append(duration)
        total_steps.append(steps)

    assert len(p1wrs) == workers, (len(p1wrs), workers)

    p1wr = np.mean(p1wrs)
    black_wr = np.mean(black_wrs)
    avg_time = np.sum(total_durations) / episodes
    avg_steps = np.sum(total_steps) / episodes

    # Load replay data
    replay = data.load_replay(args1.replay_path)

    print(f'{episodes} GAMES, {avg_time:.1f} SEC/GAME, {avg_steps:.0f} STEPS/GAME, '
          f'{100 * p1wr:.1f}% WIN({100 * black_wr:.1f}% BLACK_WIN)')

    return p1wr, black_wr, replay


def worker_play(rank, queue, barrier, args1, args2, worker_episodes):
    pi1, net1 = baselines.create_policy(args1)
    pi2, net2 = baselines.create_policy(args2)

    size = args1.size
    assert size == args2.size
    go_env = gym.make('gym_go:go-v0', size=size)

    timestart = time.time()
    p1wr, black_wr, replay, steps = self_play.play_games(go_env, pi1, pi2, worker_episodes)
    timeend = time.time()

    duration = timeend - timestart
    total_steps = sum(steps)

    queue.put((p1wr, black_wr, total_steps, duration))

    # Dump replay data into temporary file
    for i in range(barrier.parties):
        if i == rank:
            if os.path.exists(args1.replay_path):
                all_replays = data.load_replay(args1.replay_path)
                all_replays.extend(replay)
            else:
                all_replays = replay

            with open(args1.replay_path, 'wb') as f:
                pickle.dump(all_replays, f)

        barrier.wait()

def get_iter_header():
    return "TIME\tITR\tREPLAY\tC_ACC\tC_LOSS\tA_ACC\tA_LOSS\tG_LOSS\tC_WR\tR_WR\tG_WR"

def get_iter_entry(starttime, iteration, replay_len, metrics, winrates, checkpoint_pi):
    currtime = dt.now()
    delta = currtime - starttime
    iter_info = f"{str(delta).split('.')[0]}\t{iteration:02d}\t{replay_len:07d}\t"
    
    # Critic
    if metrics.crit_acc is not None and not np.isnan(metrics.crit_acc):
        iter_info += f"{100 * metrics.crit_acc:04.1f}\t{metrics.crit_loss:04.3f}\t"
    else:
        iter_info += "____\t____\t"
        
    # Actor
    if metrics.act_acc is not None and not np.isnan(metrics.act_acc):
        iter_info += f"{100 * metrics.act_acc:04.1f}\t{metrics.act_loss:04.3f}\t"
    else:
        iter_info += "____\t____\t"
        
    # Game
    if metrics.game_loss is not None and not np.isnan(metrics.game_loss):
        iter_info += f"{metrics.game_loss:04.3f}\t"
    else:
        iter_info += "____\t"
    
    iter_info += f"{100 * winrates[checkpoint_pi]:04.1f}\t{100 * winrates[baselines.RAND_PI]:04.1f}"
    return iter_info