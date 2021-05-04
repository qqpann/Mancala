from abc import ABC
from builtins import NotImplementedError


class BaseState(ABC):
    def __repr__(self):
        raise NotImplementedError()

    def clone(self):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseState):
            return NotImplemented
        raise ValueError(other)

    @property
    def sided_available_actions(self):
        raise NotImplementedError()