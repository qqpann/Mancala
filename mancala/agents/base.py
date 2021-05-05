from abc import ABC
from typing import List, Union
import numpy as np

from mancala.state.base import BaseState


class BaseAgent(ABC):
    """Abstract classs for Mancala agent"""

    def __init__(self, actions: Union[List[int], np.ndarray]):
        raise NotImplementedError

    def policy(self, state: BaseState) -> int:
        raise NotImplementedError