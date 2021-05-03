# https://github.com/openai/gym/blob/master/docs/creating-environments.md
import random
import sys
import time
from dataclasses import dataclass
from typing import List
import numpy as np
import gym
from gym import error, spaces, utils, Env
from gym.utils import seeding

from mancala.mancala import Rule, Mancala


class MancalaEnv(Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.game = Mancala()

    def step(self, action):
        pass

    def reset(self):
        self.game = Mancala()

    def render(self, mode="human"):
        self.game.render_cli_board()

    def close(self):
        pass