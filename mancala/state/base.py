from __future__ import annotations

from abc import ABC
from builtins import NotImplementedError
from typing import List, Union

import numpy as np

from mancala.rule import Rule


class BaseState(ABC):
    action_choices: List[str]
    turn: int
    must_skip: bool
    rule: Rule
    board: np.ndarray

    def __repr__(self):
        raise NotImplementedError

    def clone(self) -> BaseState:
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

    def legal_actions(self, turn: int) -> Union[List[int], None]:
        raise NotImplementedError

    def _can_continue_on_point(self, idx) -> bool:
        raise NotImplementedError

    def proceed_action(self, act: Union[int, None]) -> BaseState:
        raise NotImplementedError()

    @property
    def scores(self) -> List[int]:
        raise NotImplementedError

    @property
    def rewards(self) -> List[float]:
        raise NotImplementedError

    @property
    def _done(self) -> bool:
        raise NotImplementedError

    @property
    def perspective_boards(self) -> List[np.ndarray]:
        raise NotImplementedError
