import sys
import time
from typing import List, Union

from mancala.agents import BaseAgent, HumanAgent, RandomAgent
from mancala.mancala import MancalaEnv, turn_names


class CLIGame(object):
    def __init__(self, env: MancalaEnv, silent=False):
        self.env = env
        self.silent = silent

    def _step(self) -> None:
        if not self.silent:
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

    def play_silent(self) -> int:
        while self.env.state._winner is None:
            self._step()
        return self.env.state._winner