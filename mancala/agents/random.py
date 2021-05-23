import random
from typing import Union

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState


class RandomAgent(BaseAgent):
    """Agent with random choice policy"""

    def __init__(self, id: int, deterministic: bool = False, seed=42):
        self.id = id
        self._seed = seed
        self.deterministic = deterministic

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
        if self.deterministic:
            random.seed(self._seed)
        return random.choice(legal_actions)