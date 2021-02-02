import numpy as np


def next_idx(idx: int):
    nidx = idx + 1
    if nidx > 13:
        return 0
    return nidx


def is_ownpocket(turn: int, idx: int):
    if turn == 0:
        return idx == 6
    else:
        return idx == 13


class Mancala:
    def __init__(self):
        self.board = np.zeros((14,), dtype=np.int32)
        self.init_board()
        self.hand = 0
        self.selection = ["a", "b", "c", "d", "e", "f"]
        self.turn = 0  # player: 0, ai: 1
        self.end = False

    def init_board(self):
        self.board = np.zeros((14,), dtype=np.int32)
        for i in range(0, 6):
            self.board[i] = 4
        for i in range(7, 13):
            self.board[i] = 4

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

    def show_board(self):
        for i in range(0, 6):
            print(f"{self.board[i]:>2}", end=" ")
        print(f"[{self.board[6]:>2}]", end=" ")
        print("\n" + "--" * 7)
        for i in range(7, 13):
            print(f"{self.board[i]:>2}", end=" ")
        print(f"[{self.board[13]:>2}]", end=" ")
        print("\n" + "==" * 7)

    def show_actions(self):
        for char in self.selection:
            print(f"{char:>2}", end=" ")
        print()

    def get_player_action(self):
        key_input = input("Take one > ")
        idx = self.selection.index(key_input)
        assert idx >= 0
        return idx

    def flip_turn(self):
        self.turn = 1 if self.turn == 0 else 0

    def take_action(self, idx: int):
        self.take_pocket(idx)
        old_turn = self.turn
        self.flip_turn()
        for _ in range(self.hand):
            idx = next_idx(idx)
            print(self.hand, idx)
            if self.hand == 1 and is_ownpocket(old_turn, idx):
                self.turn = old_turn
            self.fill_pocket(idx)

    def is_ownside(self, idx: int):
        if self.turn == 0:
            return 0 <= idx < 6
        else:
            return 7 <= idx < 13

    def turn_player(self):
        self.show_actions()
        act = self.get_player_action()
        self.take_action(act)

    def turn_ai(self):
        self.end = True

    def turn_(self):
        print("turn:", self.turn)
        if self.turn == 0:
            self.turn_player()
        else:
            self.turn_ai()

    def play(self):
        while not self.end:
            self.show_board()
            self.turn_()
        print("END GAME")
        self.show_board()
