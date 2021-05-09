import sys
import time
from typing import Union, List

from mancala.agents.base import BaseAgent
from mancala.agents.human import HumanAgent
from mancala.agents.random_agent import RandomAgent
from mancala.mancala import MancalaEnv, turn_names


class CLIGame(object):
    def __init__(self, env: MancalaEnv):
        self.env = env

    def _step(self) -> None:
        print("turn:", self.env.current_agent)
        act = self.env.current_agent.policy(self.env.state)
        (next_state, reward, done) = self.env.step(act)
        self.env.state = next_state

    def play_cli(self) -> None:
        while not self.env.state._done:
            print(self.env.state)
            self.render_cli_board()
            self._step()
        print("END GAME")
        print(f"Winner: {self.env.state._winner}")
        self.render_cli_board()

    def render_cli_board(self) -> None:
        self.env.render()