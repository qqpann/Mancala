import random, sys
from typing import Union, List
import numpy as np

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState


class HumanAgent(BaseAgent):
    """Agent with random choice policy"""

    def __init__(self, actions: Union[List[int], np.ndarray, None] = None):
        # super().__init__(actions)
        pass

    def policy(self, state: BaseState) -> int:
        """
        Make a move.

        Params
        ---
        state: mancala state object

        Returns
        ---
        action: int
        """
        act = HumanAgent.get_player_action(state)
        return act

    @staticmethod
    def render_cli_actions(state: BaseState):
        print(" " * 4, end=" ")
        for char in state.action_choices:
            print(f"{char:>2}", end=" ")
        print()

    @staticmethod
    def get_player_action(state: BaseState) -> int:
        HumanAgent.render_cli_actions(state)
        while True:
            key_input = input("Take one > ")
            if key_input == "q":
                sys.exit()
            idx = state.action_choices.index(key_input)
            assert idx >= 0
            if idx in state.sided_available_actions:
                return idx
            else:
                print("Cannot pick from empty pocket")