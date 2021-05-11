import random

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState


class ValueIterationAgent(BaseAgent):
    """Agent with random choice policy"""

    def __init__(self, id: int, deterministic: bool = False, seed=42):
        # self.id = id
        # self._seed = seed
        # self.deterministic = deterministic
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
        # if self.deterministic:
        #     random.seed(self._seed)
        # return random.choice(state.legal_actions(state.current_player))
        pass