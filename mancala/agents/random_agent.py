import random

from mancala.agents.base import BaseAgent
from mancala.state.base import BaseState


class RandomAgent(BaseAgent):
    """Agent with random choice policy"""

    def __init__(self, seed=42):
        self._seed = seed
        random.seed(seed)

    def move(self, state: BaseState):
        """
        Make a move.

        Params
        ---
        state: mancala state object

        Returns
        ---
        action: int
        """
        return random.choice(state.sided_available_actions)