import random
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Rule:
    multi_lap: bool = True
    capture_opposite: bool = True
    continue_on_point: bool = True


turn_names = ["human", "ai"]


class Mancala:
    def __init__(self, pockets: int = 6, initial_stones: int = 4, rule: Rule = Rule()):
        self.__pockets = pockets
        self.__initial_stones = initial_stones
        self.rule = rule
        self.init_board()
        self.hand = 0
        self.selection = [str(i) for i in range(1, self.__pockets + 1)]
        self.turn = 0  # player: 0, ai: 1
        self.end = False

    def init_board(self):
        self.board = np.zeros(((self.__pockets + 1) * 2,), dtype=np.int32)
        # Player 1 side
        for i in range(0, self.__pockets):
            self.board[i] = self.__initial_stones
        # Player 2 side
        for i in range(self.__pockets + 1, self.__pockets * 2 + 1):
            self.board[i] = self.__initial_stones

    def take_pocket(self, idx: int):
        """
        idx: the pocket to manipulate
        num:
        """
        self.hand += self.board[idx]
        self.board[idx] = 0

    def fill_pocket(self, idx: int, num: int = 1):
        assert self.hand > 0 and num <= self.hand
        self.board[idx] += num
        self.hand -= num

    def next_idx(self, idx: int):
        return idx + 1 % self.__pockets * 2 + 1

    def opposite_idx(self, idx: int):
        assert idx <= self.__pockets * 2
        return self.__pockets * 2 - idx

    @property
    def _player0_field_range(self):
        return range(0, self.__pockets)

    @property
    def _player1_field_range(self):
        return range(self.__pockets + 1, self.__pockets * 2 + 1)

    @property
    def _player0_point_index(self):
        return self.__pockets

    @property
    def _player1_point_index(self):
        return self.__pockets * 2 + 1

    @property
    def _active_player_point_index(self):
        return (
            self._player0_point_index if self.turn == 0 else self._player1_point_index
        )

    def is_current_sided_pointpocket(self, idx: int):
        if self.turn == 0:
            return idx == self.__pockets
        else:
            return idx == self.__pockets * 2 + 1

    def is_current_sided_fieldpocket(self, idx: int):
        if self.turn == 0:
            return 0 <= idx < self.__pockets
        else:
            return self.__pockets + 1 <= idx < self.__pockets * 2 + 1

    def render_cli_board(self):
        print("\n" + "====" * (self.__pockets + 1))
        # AI side
        print(f"[{self.board[self._player1_point_index]:>2}]", end=" ")
        for i in self._player1_field_range[::-1]:
            print(f"{self.board[i]:>2}", end=" ")
        print("\n" + "----" * (self.__pockets + 1))
        # Player side
        print(" " * 4, end=" ")
        for i in self._player0_field_range:
            print(f"{self.board[i]:>2}", end=" ")
        print(f"[{self.board[self._player0_point_index]:>2}]", end=" ")
        print("\n" + "====" * (self.__pockets + 1))

    def render_cli_actions(self):
        print(" " * 4, end=" ")
        for char in self.selection:
            print(f"{char:>2}", end=" ")
        print()

    def get_sided_all_actions(self):
        if self.turn == 0:
            return list(self._player0_field_range)
        else:
            return list(self._player1_field_range)

    def filter_available_actions(self, actions: List[int]) -> List[int]:
        return [i for i in actions if self.board[i] > 0]

    def get_player_action(self):
        while True:
            key_input = input("Take one > ")
            if key_input == "q":
                sys.exit()
            idx = self.selection.index(key_input)
            assert idx >= 0
            if idx in self.filter_available_actions(self.get_sided_all_actions()):
                return idx
            else:
                print("Cannot pick from empty pocket")

    def flip_turn(self):
        print("Flip turn")
        self.turn = 1 if self.turn == 0 else 0

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

    def step_human(self):
        self.render_cli_actions()
        act = self.get_player_action()
        self.proceed_action(act)

    def step_ai(self):
        time.sleep(2)
        act = random.choice(self.filter_available_actions(self.get_sided_all_actions()))
        self.proceed_action(act)

    def _step(self):
        print("turn:", turn_names[self.turn])
        if self.turn == 0:
            self.step_human()
        else:
            self.step_ai()

    def judge_end_condition(self):
        if not self.filter_available_actions(list(self._player0_field_range)):
            self.end = True
            print("Winner:", turn_names[1])
        elif not self.filter_available_actions(list(self._player1_field_range)):
            self.end = True
            print("Winner:", turn_names[0])

    def play(self):
        while not self.end:
            self.render_cli_board()
            self._step()
            self.judge_end_condition()
        print("END GAME")
        self.render_cli_board()
