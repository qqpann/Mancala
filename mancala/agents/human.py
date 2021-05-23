from mancala.agents.negascout import negamax, negascout
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
            hint_minimax = [-float("inf")] * (2 * state.rule.pockets + 2)
            hint_negamax = [-float("inf")] * (2 * state.rule.pockets + 2)
            hint_negascout = [-float("inf")] * (2 * state.rule.pockets + 2)
            for act in legal_actions:
                v1 = minimax(state.clone().proceed_action(act), 4, self.id)
                v2 = negamax(state.clone().proceed_action(act), 4, self.id)
                v3 = negascout(state.clone().proceed_action(act), 4, self.id)
                hint_minimax[act] = v1
                hint_negamax[act] = v2
                hint_negascout[act] = v3
            start = self.id * state.rule.pockets
            print("minimax:", hint_minimax[start : start + state.rule.pockets])
            print("negamax:", hint_negamax[start : start + state.rule.pockets])
            print("negascout:", hint_negascout[start : start + state.rule.pockets])
            print()

        # Receive human input
        while True:
            key_input = input("Take from a pocket (chose the index) \n> ")
            if key_input == "q":
                sys.exit()
            if key_input not in action_choices:
                print(f"Wrong choice: {key_input}; Chose from {action_choices}")
                continue
            idx = action_choices.index(key_input)
            if self.id == 1:
                idx = state.rule.pockets * 2 - idx
            assert idx >= 0
            if idx in legal_actions:
                return idx
            else:
                print(f"Cannot pick from empty pocket: {key_input}(idx:{idx})")