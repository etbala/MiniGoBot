import numpy as np
import torch

from go_bot import actor_critic

# Random Policy
class Random:
    def __init__(self):
        self.name = 'Random'
        self.pt_model = None

    def __call__(self, go_env, **kwargs):
        """
        :param go_env:
        :param step:
        :return: Action probabilities
        """

        valid_moves = go_env.valid_moves()
        return valid_moves / np.sum(valid_moves)
    
    def __str__(self):
        return "{} {}".format(self.__class__.__name__, self.name)

RAND_PI = Random()

def create_policy(args, name=''):
    model = args.model
    size = args.size
    if model == 'ac':
        net = actor_critic.ActorCriticNet(size)
        pi = actor_critic.ActorCriticPolicy(name, net, args)
    elif model == 'rand':
        net = None
        pi = RAND_PI
        return pi, net
    else:
        raise Exception("Unknown model argument", model)

    load_weights(args, net)

    return pi, net

def load_weights(args, net):
    if args.baseline:
        assert not args.latest_checkpoint
        assert args.customdir == ''
        net.load_state_dict(torch.load(args.basepath, args.device))
    elif args.latest_checkpoint:
        assert not args.baseline
        assert args.customdir == ''
        net.load_state_dict(torch.load(args.checkpath, args.device))
    elif args.customdir != '':
        assert not args.latest_checkpoint
        assert not args.baseline
        net.load_state_dict(torch.load(args.custompath, args.device))
