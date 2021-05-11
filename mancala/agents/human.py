import random
import sys
from typing import List, Union

import numpy as np

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState


class HumanAgent(BaseAgent):
    """Agent with random choice policy"""

    def __init__(self, id: int, actions: Union[List[int], np.ndarray, None] = None):
        # super().__init__(actions)
        self.id = id

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
    def get_player_action(state: BaseState) -> int:
        # Render action choices
        print(" " * 4, end=" ")
        for char in state.action_choices:
            print(f"{char:>2}", end=" ")
        print()

        # Receive human input
        while True:
            key_input = input("Take one > ")
            if key_input == "q":
                sys.exit()
            idx = state.action_choices.index(key_input)
            assert idx >= 0
            if idx in state.legal_actions(state.current_player):
                return idx
            else:
                print("Cannot pick from empty pocket")