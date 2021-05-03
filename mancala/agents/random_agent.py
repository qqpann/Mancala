import random

from mancala.agents.base import BaseAgent
from mancala.mancala import Mancala


class RandomAgent(BaseAgent):
    """Agent with random choice policy"""

    def __init__(self, seed=42):
        self._seed = seed
        random.seed(seed)

    def move(self, game: Mancala):
        """
        Make a move.

        Params
        ---
        game: game object

        Returns
        ---
        action: int
        """
        return random.choice(game.sided_available_actions)