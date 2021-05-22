from abc import ABC
from builtins import NotImplementedError
from mancala.rule import Rule
from typing import List


class BaseState(ABC):
    action_choices: List[str]
    turn: int
    rule: Rule

    def __repr__(self):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError

    def __hash__(self):
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseState):
            return NotImplemented
        raise NotImplementedError

    @property
    def current_player(self) -> int:
        raise NotImplementedError

    def legal_actions(self, turn: int) -> List[int]:
        raise NotImplementedError

    def proceed_action(self, act: int) -> None:
        raise NotImplementedError()

    @property
    def rewards(self) -> List[float]:
        raise NotImplementedError

    def rewards_float(self, receiver_player_id: int) -> float:
        raise NotImplementedError

    def is_terminal(self) -> bool:
        raise NotImplementedError
