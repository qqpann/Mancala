from abc import ABC
from typing import List, Union

import numpy as np

from mancala.state.base import BaseState


class BaseAgent(ABC):
    """Abstract classs for Mancala agent"""

    id: int

    def __init__(self, id: int, actions: Union[List[int], np.ndarray]):
        self.id = id

    def __repr__(self):
        return f"<{self.__class__.__name__} id:{self.id}>"

    def policy(self, state: BaseState) -> Union[None, int]:
        raise NotImplementedError