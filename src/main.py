import collections
import os
import random

import gym
import numpy as np

go_env = gym.make('gym_go:go-v0', size=0)
GoVars = go_env.govars
GoGame = go_env.gogame

