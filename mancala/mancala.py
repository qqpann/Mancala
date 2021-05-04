# https://github.com/openai/gym/blob/master/docs/creating-environments.md
import random
import sys
import time
from dataclasses import dataclass
from typing import List

import gym
import numpy as np
from gym import Env, error, spaces, utils
from gym.utils import seeding

from mancala.state.base import BaseState
from mancala.agents.base import BaseAgent
from mancala.agents.random_agent import RandomAgent


@dataclass
class Rule:
    multi_lap: bool = True
    capture_opposite: bool = True
    continue_on_point: bool = True
    pockets: int = 6
    initial_stones: int = 4


turn_names = ["human", "ai"]


class MancalaState(BaseState):
    """
    Mancala State
    ---
    The board state and its utils
    """

    def __init__(self, rule: Rule):
        self.rule = rule
        self.init_board()
        self.hand = 0
        self.selection = [str(i) for i in range(1, self.rule.pockets + 1)]
        self.turn = 0  # player: 0, ai: 1
        self.end = False

    def init_board(self):
        self.board = np.zeros(((self.rule.pockets + 1) * 2,), dtype=np.int32)
        # Player 1 side
        for i in range(0, self.rule.pockets):
            self.board[i] = self.rule.initial_stones
        # Player 2 side
        for i in range(self.rule.pockets + 1, self.rule.pockets * 2 + 1):
            self.board[i] = self.rule.initial_stones

    def take_pocket(self, idx: int):
        """
        Params
        idx: index of the pocket to manipulate
        """
        self.hand += self.board[idx]
        self.board[idx] = 0

    def fill_pocket(self, idx: int, num: int = 1):
        """
        Params
        idx: index of the pocket to manipulate
        num: number of stones to fill in
        """
        assert self.hand > 0 and num <= self.hand
        print(f"[DEBUG] Fill {num} into idx:{idx} pocket")
        self.board[idx] += num
        self.hand -= num

    def next_idx(self, idx: int):
        """
        Params
        idx :int: index to check

        Returns
        :int: the next index
        """
        next_idx = (idx + 1) % ((self.rule.pockets + 1) * 2)
        return next_idx

    def opposite_idx(self, idx: int):
        """
        Params
        idx :int: index to check

        Returns
        :int: the opposide field index
        """
        assert idx <= self.rule.pockets * 2
        return self.rule.pockets * 2 - idx

    @property
    def _player0_field_range(self):
        return range(0, self.rule.pockets)

    @property
    def _player1_field_range(self):
        return range(self.rule.pockets + 1, self.rule.pockets * 2 + 1)

    @property
    def _player0_point_index(self):
        return self.rule.pockets

    @property
    def _player1_point_index(self):
        return self.rule.pockets * 2 + 1

    @property
    def _active_player_point_index(self):
        return (
            self._player0_point_index if self.turn == 0 else self._player1_point_index
        )

    def is_current_sided_pointpocket(self, idx: int):
        if self.turn == 0:
            return idx == self.rule.pockets
        else:
            return idx == self.rule.pockets * 2 + 1

    def is_current_sided_fieldpocket(self, idx: int):
        if self.turn == 0:
            return 0 <= idx < self.rule.pockets
        else:
            return self.rule.pockets + 1 <= idx < self.rule.pockets * 2 + 1

    @property
    def sided_all_actions(self):
        if self.turn == 0:
            return list(self._player0_field_range)
        else:
            return list(self._player1_field_range)

    def filter_available_actions(self, actions: List[int]) -> List[int]:
        return [i for i in actions if self.board[i] > 0]

    @property
    def sided_available_actions(self):
        return self.filter_available_actions(self.sided_all_actions)

    def proceed_action(self, idx: int):
        self.take_pocket(idx)
        continue_turn = False
        for _ in range(self.hand):
            idx = self.next_idx(idx)
            if (
                self.hand == 1
                and self.rule.continue_on_point
                and self.is_current_sided_pointpocket(idx)
            ):
                continue_turn = True
            if (
                self.hand == 1
                and self.rule.capture_opposite
                and self.is_current_sided_fieldpocket(idx)
                and self.board[idx] == 0
                and self.board[self.opposite_idx(idx)] > 0
            ):
                self.take_pocket(self.opposite_idx(idx))
                self.fill_pocket(self._active_player_point_index, self.hand)
                break
            self.fill_pocket(idx)
        if not (continue_turn and self.rule.multi_lap):
            self.flip_turn()

    def flip_turn(self):
        self.turn = 1 if self.turn == 0 else 0


class MancalaEnv(Env):
    metadata = {"render.modes": ["human"]}

    # Common Env functions
    # --------------------
    def __init__(self, agent: BaseAgent):
        super().__init__()
        self.agent = agent
        self.rule = Rule()
        self.state = MancalaState(self.rule)

    def step(self, action):
        pass

    def reset(self):
        self.state = MancalaState(Rule())

    def render(self, mode="human"):
        self.render_cli_board()

    def close(self):
        pass

    # CLI functions
    # -------------
    def step_human(self):
        self.render_cli_actions()
        act = self.get_player_action()
        self.state.proceed_action(act)

    def step_ai(self):
        time.sleep(2)
        act = self.agent.move(self.state)
        print(f"AI's turn. moving {act}")
        self.state.proceed_action(act)

    def _step(self):
        print("turn:", turn_names[self.state.turn])
        if self.state.turn == 0:
            self.step_human()
        else:
            self.step_ai()

    def judge_end_condition(self):
        if not self.state.filter_available_actions(
            list(self.state._player0_field_range)
        ):
            self.state.end = True
            print("Winner:", turn_names[1])
        elif not self.state.filter_available_actions(
            list(self.state._player1_field_range)
        ):
            self.state.end = True
            print("Winner:", turn_names[0])

    def play_cli(self):
        while not self.state.end:
            self.render_cli_board()
            self._step()
            self.judge_end_condition()
        print("END GAME")
        self.render_cli_board()

    def get_player_action(self):
        while True:
            key_input = input("Take one > ")
            if key_input == "q":
                sys.exit()
            idx = self.state.selection.index(key_input)
            assert idx >= 0
            if idx in self.state.sided_available_actions:
                return idx
            else:
                print("Cannot pick from empty pocket")

    def render_cli_board(self):
        print("\n" + "====" * (self.rule.pockets + 1))
        # AI side
        print(f"[{self.state.board[self.state._player1_point_index]:>2}]", end=" ")
        for i in self.state._player1_field_range[::-1]:
            print(f"{self.state.board[i]:>2}", end=" ")
        print("\n" + "----" * (self.rule.pockets + 1))
        # Player side
        print(" " * 4, end=" ")
        for i in self.state._player0_field_range:
            print(f"{self.state.board[i]:>2}", end=" ")
        print(f"[{self.state.board[self.state._player0_point_index]:>2}]", end=" ")
        print("\n" + "====" * (self.rule.pockets + 1))

    def render_cli_actions(self):
        print(" " * 4, end=" ")
        for char in self.state.selection:
            print(f"{char:>2}", end=" ")
        print()