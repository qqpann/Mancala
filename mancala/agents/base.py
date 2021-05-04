from abc import ABC

from mancala.state.base import BaseState


class BaseAgent(ABC):
    """Abstract classs for Mancala agent"""

    def policy(self, state: BaseState) -> int:
        raise NotImplementedError