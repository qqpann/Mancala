# https://github.com/openai/gym/blob/master/docs/creating-environments.md
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class MancalaEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        pass