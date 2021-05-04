import time
import sys

from mancala.agents.base import BaseAgent
from mancala.mancala import MancalaEnv, turn_names


class CLIGame(object):
    def __init__(self, agent: BaseAgent, env: MancalaEnv):
        self.agent = agent
        self.env = env

    def step_human(self):
        act = self.get_player_action()
        self.env.state.proceed_action(act)

    def step_ai(self):
        time.sleep(2)
        act = self.agent.policy(self.env.state)
        print(f"AI's turn. moving {act}")
        self.env.state.proceed_action(act)

    def _step(self):
        print("turn:", turn_names[self.env.state.turn])
        if self.env.state.turn == 0:
            self.step_human()
        else:
            self.step_ai()

    def judge_end_condition(self):
        if not self.env.state.filter_available_actions(
            list(self.env.state._player0_field_range)
        ):
            self.env.state.end = True
            print("Winner:", turn_names[1])
        elif not self.env.state.filter_available_actions(
            list(self.env.state._player1_field_range)
        ):
            self.env.state.end = True
            print("Winner:", turn_names[0])

    def play_cli(self):
        while not self.env.state.end:
            self.render_cli_board()
            self._step()
            self.judge_end_condition()
        print("END GAME")
        self.render_cli_board()

    def render_cli_actions(self):
        print(" " * 4, end=" ")
        for char in self.env.state.selection:
            print(f"{char:>2}", end=" ")
        print()

    def get_player_action(self) -> int:
        self.render_cli_actions()
        while True:
            key_input = input("Take one > ")
            if key_input == "q":
                sys.exit()
            idx = self.env.state.selection.index(key_input)
            assert idx >= 0
            if idx in self.env.state.sided_available_actions:
                return idx
            else:
                print("Cannot pick from empty pocket")

    def render_cli_board(self):
        print("\n" + "====" * (self.env.rule.pockets + 1))
        # AI side
        print(
            f"[{self.env.state.board[self.env.state._player1_point_index]:>2}]", end=" "
        )
        for i in self.env.state._player1_field_range[::-1]:
            print(f"{self.env.state.board[i]:>2}", end=" ")
        print("\n" + "----" * (self.env.rule.pockets + 1))
        # Player side
        print(" " * 4, end=" ")
        for i in self.env.state._player0_field_range:
            print(f"{self.env.state.board[i]:>2}", end=" ")
        print(
            f"[{self.env.state.board[self.env.state._player0_point_index]:>2}]", end=" "
        )
        print("\n" + "====" * (self.env.rule.pockets + 1))