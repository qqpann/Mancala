import random

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState


class RandomAgent(BaseAgent):
    """Agent with random choice policy"""

    def __init__(self, deterministic: bool = False, seed=42):
        self._seed = seed
        self.deterministic = deterministic

    def policy(self, state: BaseState):
        """
        Make a move.

        Params
        ---
        state: mancala state object

        Returns
        ---
        action: int
        """
        if self.deterministic:
            random.seed(self._seed)
        return random.choice(state.sided_available_actions)