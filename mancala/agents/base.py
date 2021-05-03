from abc import ABC


class BaseAgent(ABC):
    """Abstract classs for Mancala agent"""

    def move(self, game) -> int:
        raise NotImplementedError()