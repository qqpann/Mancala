from mancala.agents.minimax import minimax
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
        self.hint = True

    def policy(self, state: BaseState) -> Union[int, None]:
        """
        Make a move.

        Params
        ---
        state: mancala state object

        Returns
        ---
        action: int
        """
        legal_actions = state.legal_actions(state.current_player)
        if legal_actions is None:
            return None
        act = self.get_player_action(state, legal_actions)
        return act

    def get_player_action(self, state: BaseState, legal_actions: List[int]) -> int:
        # Render action choices
        print(" " * 4, end=" ")
        action_choices = [str(i) for i in range(1, state.rule.pockets + 1)]
        for i in action_choices:
            print(f"{i:>2}", end=" ")
        print()
        if self.hint:
            for act in range(0, 6):
                v = minimax(state.clone().proceed_action(act), 4, 0)
                print(f"{v}", end=" ")
            print()

        # Receive human input
        while True:
            key_input = input("Take from a pocket (chose the index) \n> ")
            if key_input == "q":
                sys.exit()
            idx = action_choices.index(key_input)
            if self.id == 1:
                idx = state.rule.pockets * 2 - idx
            assert idx >= 0
            if idx in legal_actions:
                return idx
            else:
                print("Cannot pick from empty pocket:", idx)