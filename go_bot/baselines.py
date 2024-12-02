import numpy as np
import torch

from go_bot import actor_critic

class Human:
    def __init__(self, render):
        self.name = 'Human'
        self.temp = None
        self.render = render

    def __call__(self, go_env, **kwargs):
        state = go_env.get_state()
        valid_moves = go_env.valid_moves()

        # Human interface
        if self.render == 'human':
            while True:
                player_action = go_env.render(self.render)
                if player_action is None:
                    player_action = go_env.action_space.n - 1
                else:
                    player_action = data.GoGame.action_2d_to_1d(player_action, state)
                if valid_moves[player_action] > 0:
                    break
        else:
            while True:
                try:
                    go_env.render(self.render)
                    coor = input("Enter actions coordinates i j:\n")
                    if coor == 'p':
                        player_action = None
                    elif coor == 'e':
                        player_action = None
                        exit()
                    else:
                        coor = coor.split(' ')
                        player_action = (int(coor[0]), int(coor[1]))

                    player_action = data.GoGame.action_2d_to_1d(player_action, state)
                    if valid_moves[player_action] > 0:
                        break
                except Exception:
                    pass

        action_probs = np.zeros(data.GoGame.action_size(state))
        action_probs[player_action] = 1

        return action_probs
    
    def __str__(self):
        return "{} {}".format(self.__class__.__name__, self.name)

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
HUMAN_PI = Human('terminal')

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
    elif model == 'human':
        net = None
        pi = Human(args.render)
        return pi, net
    else:
        raise Exception("Unknown model argument", model)

    load_weights(args, net)

    return pi, net

def load_weights(args, net):
    if args.latest_checkpoint:
        assert args.customdir == ''
        net.load_state_dict(torch.load(args.checkpath, args.device, weights_only=True))
    elif args.customdir != '':
        assert not args.latest_checkpoint
        net.load_state_dict(torch.load(args.custompath, args.device, weights_only=True))
