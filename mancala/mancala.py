import numpy as np
import sys
import random
from dataclasses import dataclass


def next_idx(idx: int):
    nidx = idx + 1
    if nidx > 13:
        return 0
    return nidx


@dataclass
class Rule:
    multi_lap: bool = True


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
        self.hand = self.board[idx]
        self.board[idx] = 0

    def fill_pocket(self, idx: int):
        if self.hand <= 0:
            return
        self.board[idx] += 1
        self.hand -= 1

    def is_own_pointpocket(self, idx: int):
        if self.turn == 0:
            return idx == 6
        else:
            return idx == 13

    def is_own_fieldpocket(self, idx: int):
        if self.turn == 0:
            return 0 <= idx < 6
        else:
            return 7 <= idx < 13

    def render_cli(self):
        print("\n" + "====" * (self.__pockets + 1))
        # AI side
        print(f"[{self.board[self.__pockets*2+1]:>2}]", end=" ")
        for i in range(self.__pockets + 1, self.__pockets * 2 + 1)[::-1]:
            print(f"{self.board[i]:>2}", end=" ")
        print("\n" + "----" * (self.__pockets + 1))
        # Player side
        print(" " * 4, end=" ")
        for i in range(0, self.__pockets):
            print(f"{self.board[i]:>2}", end=" ")
        print(f"[{self.board[self.__pockets]:>2}]", end=" ")
        print("\n" + "====" * (self.__pockets + 1))

    def show_actions(self):
        print(" " * 4, end=" ")
        for char in self.selection:
            print(f"{char:>2}", end=" ")
        print()

    def get_all_actions(self):
        if self.turn == 0:
            return list(range(0, 7))
        else:
            return list(range(7, 14))

    def get_available_actions(self):
        return [i for i in self.get_all_actions() if self.board[i] > 0]

    def get_player_action(self):
        while True:
            key_input = input("Take one > ")
            if key_input == "q":
                sys.exit()
            idx = self.selection.index(key_input)
            assert idx >= 0
            if idx in self.get_available_actions():
                return idx
            else:
                print("Cannot pick from empty pocket")

    def flip_turn(self):
        self.turn = 1 if self.turn == 0 else 0

    def take_action(self, idx: int):
        self.take_pocket(idx)
        current_turn = self.turn
        continue_turn = False
        for _ in range(self.hand):
            idx = next_idx(idx)
            print(self.hand, idx)
            if self.hand == 1 and self.is_own_pointpocket(idx):
                continue_turn = True
            self.fill_pocket(idx)
        if not (continue_turn and self.rule.multi_lap):
            self.flip_turn()

    def step_human(self):
        self.show_actions()
        act = self.get_player_action()
        self.take_action(act)

    def step_ai(self):
        act = random.choice(self.get_available_actions())
        self.take_action(act)

    def _step(self):
        print("turn:", self.turn)
        if self.turn == 0:
            self.step_human()
        else:
            self.step_ai()

    def play(self):
        while not self.end:
            self.render_cli()
            self._step()
        print("END GAME")
        self.render_cli()
