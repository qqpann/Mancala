from abc import ABC
from builtins import NotImplementedError


class BaseState(ABC):
    @property
    def sided_available_actions(self):
        raise NotImplementedError()